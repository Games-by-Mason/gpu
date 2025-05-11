//! The frontend of the API. This is what you call into to render your graphics.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.gpu);
const writers = @import("writers.zig");
const OwnedWriterVolatile = writers.OwnedWriterVolatile;
const tracy = @import("tracy");
const CpuZone = tracy.Zone;
const TracyQueue = tracy.GpuQueue;
const gpu = @import("root.zig");
const global_options = @import("root.zig").options;
const builtin = @import("builtin");
const btypes = @import("btypes.zig");

const Ctx = @This();

const Backend = global_options.Backend;

const tracy_gpu_pool = "gpu";

pub const Extent2D = struct {
    width: u32,
    height: u32,
};

pub const Offset2D = struct {
    x: i32,
    y: i32,
};

pub const Rect2D = struct {
    offset: Offset2D,
    extent: Extent2D,
};

pub const Viewport = struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
};

pub const MemoryRequirements = struct {
    const DedicatedAllocationAffinity = enum {
        discouraged,
        preferred,
        required,
    };
    size: u64,
    alignment: u64,
    dedicated_allocation: DedicatedAllocationAffinity,

    /// Bumps the given offset by these memory requirements. If a dedicated allocation is preferred
    /// or required, the offset is left unchanged.
    pub fn bump(self: @This(), offset: *u64) void {
        if (self.dedicated_allocation != .discouraged) return;
        offset.* = std.mem.alignForward(u64, offset.*, self.alignment);
        offset.* += self.size;
    }
};

pub const DebugName = struct {
    str: [*:0]const u8,
    index: ?usize = null,
};

pub const Device = struct {
    // https://registry.khronos.org/vulkan/specs/1.3/html/chap33.html#limits-minmax
    pub const max_uniform_buf_offset_alignment = 256;
    // https://registry.khronos.org/vulkan/specs/1.3/html/chap33.html#limits-minmax
    pub const max_storage_buf_offset_alignment = 256;

    pub const Kind = enum {
        other,
        integrated,
        discrete,
        virtual,
        cpu,
    };

    kind: Kind,
    uniform_buf_offset_alignment: u16,
    storage_buf_offset_alignment: u16,
    timestamp_period: f32,
    tracy_queue: TracyQueue,
    surface_format: ImageFormat,
};

backend: Backend,

device: Device,

/// The number of frames that can be in flight at once.
frames_in_flight: u4,
/// The current frame in flight.
frame: u8 = 0,
in_frame: bool = false,

max_alignment: bool,

timestamp_queries: bool,
tracy_queries: [global_options.max_frames_in_flight]u16 = @splat(0),
validate: bool,

pub const DebugMode = enum(u8) {
    /// Enables graphics API validation and debug output. High performance cost.
    ///
    /// Will emit warning if not available on host.
    validate = 2,
    /// Enables debug output. Minimal to no performance cost, may aid profiling software in
    /// providing readable output.
    ///
    /// Will emit warning if not available on host.
    output = 1,
    /// No debugging support.
    none = 0,

    pub fn gte(lhs: @This(), rhs: @This()) bool {
        return @intFromEnum(lhs) >= @intFromEnum(rhs);
    }
};

pub const Options = struct {
    pub const default_device_type_ranks = b: {
        var ranks = std.EnumArray(Device.Kind, u8).initFill(0);
        ranks.set(.discrete, 2);
        ranks.set(.integrated, 1);
        break :b ranks;
    };

    pub const Version = struct {
        major: u7,
        minor: u10,
        patch: u12,
    };

    gpa: Allocator,
    application_name: ?[:0]const u8 = null,
    application_version: Version = .{
        .major = 0,
        .minor = 0,
        .patch = 0,
    },
    engine_name: ?[:0]const u8,
    engine_version: Version = .{
        .major = 0,
        .minor = 0,
        .patch = 0,
    },
    frames_in_flight: u4,
    framebuf_extent: Extent2D,
    backend: Backend.Options,
    device_type_ranks: std.EnumArray(Device.Kind, u8) = default_device_type_ranks,
    timestamp_queries: bool,
    debug: DebugMode = if (builtin.mode == .Debug) .validate else .none,
    /// Disables potentially problematic features. For example, disables all implicit layers in
    /// Vulkan. This may disrupt functionality expected by the user and should only be enabled
    /// when a problem occurs.
    safe_mode: bool = false,
    max_alignment: bool = false,
};

pub fn init(options: Options) @This() {
    const zone = tracy.Zone.begin(.{ .name = "gpu init", .src = @src() });
    defer zone.end();
    log.debug("Initializing GPU frontend", .{});

    assert(options.frames_in_flight > 0);
    assert(options.frames_in_flight <= global_options.max_frames_in_flight);

    var gx: @This() = .{
        .backend = undefined,
        .device = undefined,
        .frames_in_flight = options.frames_in_flight,
        .max_alignment = options.max_alignment,
        .timestamp_queries = options.timestamp_queries,
        .validate = options.debug.gte(.validate),
    };

    Backend.init(&gx, options);

    gx.device = Backend.getDevice(&gx);

    if (gx.max_alignment) {
        gx.device.uniform_buf_offset_alignment = Device.max_uniform_buf_offset_alignment;
        gx.device.storage_buf_offset_alignment = Device.max_storage_buf_offset_alignment;
    }

    return gx;
}

/// Destroys the context. Must not be in use, see `waitIdle`.
pub fn deinit(self: *@This(), gpa: Allocator) void {
    Backend.deinit(self, gpa);
    self.* = undefined;
}

/// Gets a pointer to the backend of the current type, or emits a compiler error if the type does
/// not match.
pub inline fn getBackend(self: *@This(), T: type) *T {
    assert(global_options.Backend == T);
    return &self.backend;
}

/// See `getBackend`.
pub inline fn getBackendConst(self: *const @This(), T: type) *const T {
    assert(global_options.Backend == T);
    return &self.backend;
}

pub fn DedicatedBuf(kind: BufKind) type {
    return struct {
        memory: MemoryUnsized,
        buf: Buf(kind),

        pub const Options = struct {
            name: DebugName,
            size: u64,
        };

        pub inline fn init(
            gx: *Ctx,
            options: @This().Options,
        ) DedicatedBuf(kind) {
            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();
            comptime kind.assertNonZero();
            assert(options.size > 0); // Vulkan doesn't support zero sized buffers
            const result = Backend.dedicatedBufCreate(gx, options.name, kind, options.size);
            tracy.alloc(.{
                .ptr = @ptrFromInt(@intFromEnum(result.dedicated.memory)),
                .size = result.size,
                .pool_name = tracy_gpu_pool,
            });
            return .{
                .memory = @enumFromInt(@intFromEnum(result.dedicated.memory)),
                .buf = @enumFromInt(@intFromEnum(result.dedicated.buf)),
            };
        }

        pub fn deinit(self: @This(), gx: *Ctx) void {
            Backend.bufDestroy(gx, self.buf.as(.{}));
            self.memory.deinit(gx);
        }

        pub inline fn as(self: @This(), comptime result_kind: BufKind) DedicatedBuf(result_kind) {
            return .{
                .memory = self.memory.as(.{ .usage = .{ .buf = result_kind } }),
                .buf = self.buf.as(result_kind),
            };
        }
    };
}

pub fn DedicatedReadbackBuf(kind: BufKind) type {
    return struct {
        memory: MemoryUnsized,
        buf: Buf(kind),
        data: []const u8,

        pub const Options = struct {
            name: DebugName,
            size: u64,
        };

        pub inline fn init(
            gx: *Ctx,
            options: @This().Options,
        ) @This() {
            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();
            comptime kind.assertNonZero();
            const result = Backend.dedicatedReadbackBufCreate(gx, options.name, kind, options.size);
            tracy.alloc(.{
                .ptr = @ptrFromInt(@intFromEnum(result.dedicated.memory)),
                .size = result.size,
                .pool_name = tracy_gpu_pool,
            });
            return .{
                .memory = @enumFromInt(@intFromEnum(result.dedicated.memory)),
                .buf = @enumFromInt(@intFromEnum(result.dedicated.buf)),
                .data = result.dedicated.data,
            };
        }

        pub fn deinit(self: @This(), gx: *Ctx) void {
            Backend.bufDestroy(gx, self.buf.as(.{}));
            self.memory.deinit(gx);
        }

        pub inline fn as(self: @This(), comptime result_kind: BufKind) DedicatedReadbackBuf(result_kind) {
            return .{
                .memory = self.memory.as(.{ .access = .read, .usage = .{ .buf = result_kind } }),
                .buf = self.buf.as(result_kind),
                .ptr = self.ptr,
                .size = self.size,
            };
        }

        pub inline fn asDedicated(self: @This(), comptime result_kind: BufKind) DedicatedBuf(result_kind) {
            return .{
                .memory = self.memory.as(.{ .usage = .{ .buf = result_kind } }),
                .buf = self.buf.as(result_kind),
            };
        }
    };
}

pub fn DedicatedUploadBuf(kind: BufKind) type {
    return struct {
        memory: MemoryUnsized,
        buf: Buf(kind),
        data: []volatile anyopaque,

        pub const Options = struct {
            name: DebugName,
            size: u64,
            prefer_device_local: bool,
        };

        pub inline fn init(
            gx: *Ctx,
            options: @This().Options,
        ) DedicatedUploadBuf(kind) {
            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();
            comptime kind.assertNonZero();
            const result = Backend.dedicatedUploadBufCreate(
                gx,
                options.name,
                kind,
                options.size,
                options.prefer_device_local,
            );
            tracy.alloc(.{
                .ptr = @ptrFromInt(@intFromEnum(result.dedicated.memory)),
                .size = result.size,
                .pool_name = tracy_gpu_pool,
            });
            return .{
                .memory = @enumFromInt(@intFromEnum(result.dedicated.memory)),
                .buf = @enumFromInt(@intFromEnum(result.dedicated.buf)),
                .data = result.dedicated.data,
            };
        }

        pub fn deinit(self: @This(), gx: *Ctx) void {
            Backend.bufDestroy(gx, self.buf.as(.{}));
            self.memory.deinit(gx);
        }

        pub inline fn as(self: @This(), comptime result_kind: BufKind) DedicatedUploadBuf(result_kind) {
            return .{
                .memory = self.memory.as(.{ .access = .write, .kind = .{ .buf = result_kind } }),
                .buf = self.buf.as(result_kind),
                .data = self.data,
            };
        }

        pub inline fn asDedicated(self: @This(), comptime result_kind: BufKind) DedicatedBuf(result_kind) {
            return .{
                .memory = self.memory.as(.{ .usage = .{ .buf = result_kind } }),
                .buf = self.buf.as(result_kind),
            };
        }

        pub const WriterOptions = struct {
            offset: u64 = 0,
            size: ?u64 = null,
        };

        pub fn writer(self: @This(), options: WriterOptions) OwnedWriterVolatile {
            return (OwnedWriterVolatile{
                .write_only_memory = self.data.ptr,
                .pos = 0,
                .size = self.data.len,
            }).spliced(options.offset, options.size);
        }
    };
}

pub fn DedicatedAllocation(Dedicated: type) type {
    return struct {
        dedicated: Dedicated,
        size: u64,
    };
}

pub const ImageTransition = extern struct {
    backend: Backend.ImageTransition,

    pub const Range = struct {
        aspect: ImageAspect,
        base_mip_level: u32 = 0,
        mip_level_count: u32 = 1,
        base_array_layer: u32 = 0,
        array_layer_count: u32 = 1,
    };

    pub const UndefinedToTransferDstOptions = struct {
        handle: ImageHandle,
        range: Range,
    };

    pub fn undefinedToTransferDst(options: UndefinedToTransferDstOptions) @This() {
        return Backend.imageTransitionUndefinedToTransferDst(options);
    }

    pub const UndefinedToColorAttachmentOptions = struct {
        handle: ImageHandle,
        range: Range,
    };

    pub fn undefinedToColorAttachment(options: UndefinedToColorAttachmentOptions) @This() {
        return Backend.imageTransitionUndefinedToColorAttachment(options);
    }

    pub const UndefinedToColorAttachmentOptionsAfterRead = struct {
        pub const Stage = packed struct {
            vertex_shader: bool = false,
            fragment_shader: bool = false,
            compute_shader: bool = false,
        };

        handle: ImageHandle,
        range: Range,
        src_stage: Stage,
    };

    pub fn undefinedToColorAttachmentAfterRead(options: UndefinedToColorAttachmentOptionsAfterRead) @This() {
        return Backend.imageTransitionUndefinedToColorAttachmentAfterRead(options);
    }

    pub const TransferDstToReadOnlyOptions = struct {
        pub const Stage = packed struct {
            vertex_shader: bool = false,
            fragment_shader: bool = false,
            compute_shader: bool = false,
        };

        handle: ImageHandle,
        range: Range,
        dst_stage: Stage,
    };

    pub fn transferDstToReadOnly(options: TransferDstToReadOnlyOptions) @This() {
        return Backend.imageTransitionTransferDstToReadOnly(options);
    }

    pub const TransferDstToColorAttachmentOptions = struct {
        handle: ImageHandle,
        range: Range,
    };

    pub fn transferDstToColorAttachment(options: TransferDstToColorAttachmentOptions) @This() {
        return Backend.imageTransitionTransferDstToColorAttachment(options);
    }

    pub const ReadOnlyToColorAttachmentOptions = struct {
        pub const Stage = packed struct {
            vertex_shader: bool = false,
            fragment_shader: bool = false,
            compute_shader: bool = false,
        };

        handle: ImageHandle,
        range: Range,
        src_stage: Stage,
    };

    pub fn readOnlyToColorAttachment(options: ReadOnlyToColorAttachmentOptions) @This() {
        return Backend.imageTransitionReadOnlyToColorAttachment(options);
    }

    pub const ColorAttachmentToReadOnlyOptions = struct {
        pub const Stage = packed struct {
            vertex_shader: bool = false,
            fragment_shader: bool = false,
            compute_shader: bool = false,
        };

        handle: ImageHandle,
        range: Range,
        dst_stage: Stage,
    };

    pub fn colorAttachmentToReadOnly(options: ColorAttachmentToReadOnlyOptions) @This() {
        return Backend.imageTransitionColorAttachmentToReadOnly(options);
    }

    pub const asBackendSlice = AsBackendSlice(@This()).mixin;
};

pub const ImageUpload = struct {
    pub const Region = extern struct {
        pub const Options = struct {
            aspect: ImageAspect,
            buffer_offset: u64 = 0,
            buffer_row_length: ?u32 = null,
            buffer_image_height: ?u32 = null,
            mip_level: u32 = 0,
            base_array_layer: u32 = 0,
            array_layer_count: u32 = 1,
            image_offset: Offset = .{ .x = 0, .y = 0, .z = 0 },
            image_extent: ImageExtent,
        };

        backend: Backend.ImageUploadRegion,

        pub fn init(options: @This().Options) @This() {
            return Backend.imageUploadRegionInit(options);
        }

        pub const asBackendSlice = AsBackendSlice(@This()).mixin;
    };

    pub const Offset = struct { x: i32, y: i32, z: i32 };

    dst: ImageHandle,
    src: Buf(.{ .transfer_src = true }),
    base_mip_level: u32 = 0,
    mip_level_count: u32 = 1,
    regions: []const Region,
};

pub const BufferUpload = struct {
    pub const Region = extern struct {
        pub const Options = struct {
            src_offset: u64 = 0,
            dst_offset: u64 = 0,
            size: u64,
        };

        backend: Backend.BufferUploadRegion,

        pub fn init(options: @This().Options) @This() {
            return Backend.bufferUploadRegionInit(options);
        }

        pub const asBackendSlice = AsBackendSlice(@This()).mixin;
    };

    dst: Buf(.{ .transfer_dst = true }),
    src: Buf(.{ .transfer_src = true }),
    regions: []const Region,
};

pub const Attachment = struct {
    backend: Backend.Attachment,

    const LoadOp = union(enum) {
        load: void,
        clear_color: [4]f32,
        dont_care: void,
    };

    pub const Layout = enum {};

    // Resolve options not currently supported through the public interface, some thought is needed
    // to make this compatible with the DX12 style API. Store op is also assumed to be store for
    // now.
    pub const Options = struct {
        view: ImageView,
        load_op: LoadOp,
    };

    pub fn init(options: @This().Options) @This() {
        return Backend.attachmentInit(options);
    }

    pub const asBackendSlice = AsBackendSlice(@This()).mixin;
};

pub const CmdBuf = enum(u64) {
    _,

    /// A unique ID used for Tracy queries.
    pub const TracyQueryId = packed struct(u16) {
        pub const cap = std.math.maxInt(@FieldType(@This(), "index"));
        frame: u8,
        index: u8,

        /// Returns the next available query ID for this frame, or panics if there are none left.
        pub fn next(gx: *Ctx) @This() {
            if (gx.tracy_queries[gx.frame] > TracyQueryId.cap) @panic("out of Tracy queries");
            const result: @This() = .{
                .index = @intCast(gx.tracy_queries[gx.frame]),
                .frame = gx.frame,
            };
            gx.tracy_queries[gx.frame] += 1;
            return result;
        }
    };

    pub fn init(
        gx: *Ctx,
        comptime loc: tracy.SourceLocation.InitOptions,
    ) @This() {
        return .initFromPtr(gx, .init(loc));
    }

    pub fn initFromPtr(
        gx: *Ctx,
        loc: *const tracy.SourceLocation,
    ) @This() {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();
        assert(gx.in_frame);
        const cb = Backend.cmdBufCreate(gx, loc);
        cb.beginZoneFromPtr(gx, loc);
        return cb;
    }

    pub const BeginRenderingOptions = struct {
        color_attachments: []const Attachment = &.{},
        depth_attachment: ?*Attachment = null,
        stencil_attachment: ?*Attachment = null,
        area: Rect2D,
        viewport: Viewport,
        scissor: Rect2D,
    };

    pub fn beginRendering(self: @This(), gx: *Ctx, options: BeginRenderingOptions) void {
        Backend.cmdBufBeginRendering(gx, self, options);
        Backend.cmdBufSetViewport(gx, self, options.viewport);
        Backend.cmdBufSetScissor(gx, self, options.scissor);
    }

    pub fn endRendering(self: @This(), gx: *Ctx) void {
        Backend.cmdBufEndRendering(gx, self);
    }

    pub fn setViewport(self: @This(), gx: *Ctx, viewport: Viewport) void {
        Backend.cmdBufSetViewport(gx, self, viewport);
    }

    pub fn setScissor(self: @This(), gx: *Ctx, scissor: Extent2D) void {
        Backend.cmdBufSetScissor(gx, self, scissor);
    }

    pub fn submit(self: @This(), gx: *Ctx) void {
        const zone = CpuZone.begin(.{ .src = @src() });
        defer zone.end();
        assert(gx.in_frame);
        Backend.cmdBufSubmit(gx, self);
    }

    pub fn bindPipeline(self: @This(), gx: *Ctx, pipeline: Pipeline) void {
        const zone = CpuZone.begin(.{ .src = @src() });
        defer zone.end();
        Backend.cmdBufBindPipeline(gx, self, pipeline);
    }

    pub fn bindDescSet(
        self: @This(),
        gx: *Ctx,
        pipeline: Pipeline,
        set: DescSet,
    ) void {
        const zone = CpuZone.begin(.{ .src = @src() });
        defer zone.end();
        Backend.cmdBufBindDescSet(gx, self, pipeline, set);
    }

    pub const DrawOptions = struct {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    };

    pub fn draw(self: @This(), gx: *Ctx, options: DrawOptions) void {
        const zone = CpuZone.begin(.{ .src = @src() });
        defer zone.end();
        Backend.cmdBufDraw(gx, self, options);
    }

    pub fn transitionImages(self: @This(), gx: *Ctx, transitions: []const ImageTransition) void {
        Backend.cmdBufTransitionImages(gx, self, transitions);
    }

    pub fn uploadImage(self: @This(), gx: *Ctx, options: ImageUpload) void {
        Backend.cmdBufUploadImage(
            gx,
            self,
            options.dst,
            options.src.as(.{}),
            options.regions,
        );
    }

    // XXX: dead code right now, some errors?
    pub fn uploadBuffer(self: @This(), gx: *Ctx, options: BufferUpload) void {
        Backend.cmdBufUploadBuffer(
            gx,
            self.asUntyped(),
            options.dst.as(.{}),
            options.src.as(.{}),
            options.regions,
        );
    }

    pub fn beginZone(self: @This(), gx: *Ctx, comptime loc: tracy.SourceLocation.InitOptions) void {
        self.beginZoneFromPtr(gx, .init(loc));
    }

    pub fn beginZoneFromPtr(self: @This(), gx: *Ctx, loc: *const tracy.SourceLocation) void {
        Backend.cmdBufBeginZone(gx, self, loc);
    }

    pub fn endZone(self: @This(), gx: *Ctx) void {
        Backend.cmdBufEndZone(gx, self);
    }

    pub inline fn fromBackendType(value: Backend.CmdBuf) @This() {
        comptime assert(@sizeOf(Backend.CmdBuf) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.CmdBuf {
        comptime assert(@sizeOf(Backend.CmdBuf) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }
};

pub const EndFrameOptions = struct {
    /// If true, presents the current image, otherwise ends the frame without presentation.
    ///
    /// Some drivers sometimes block here waiting for the swapchain image instead of when acquiring
    /// it.
    present: bool = true,
};

/// Ends the current frame.
pub fn endFrame(self: *@This(), options: EndFrameOptions) void {
    const zone = CpuZone.begin(.{ .src = @src() });
    defer zone.end();
    const blocking_zone = CpuZone.begin(.{
        .src = @src(),
        .color = global_options.blocking_zone_color,
    });
    defer blocking_zone.end();
    Backend.endFrame(self, options);
    const Frame = @TypeOf(self.frame);
    const FramesInFlight = @TypeOf(self.frames_in_flight);
    comptime assert(std.math.maxInt(FramesInFlight) < std.math.maxInt(Frame));
    self.frame = (self.frame + 1) % self.frames_in_flight;
    assert(self.in_frame);
    self.in_frame = false;
}

/// Acquires the next swapchain image, blocking until it's available if necessary. Returns the
/// nanoseconds spent blocking.
///
/// Returns null if the swapchain needed to be recreated, in which case you should drop this frame.
pub fn acquireNextImage(self: *@This(), framebuf_extent: Ctx.Extent2D) ImageView.Sized2D {
    const zone = CpuZone.begin(.{
        .src = @src(),
        .color = global_options.blocking_zone_color,
    });
    defer zone.end();
    assert(framebuf_extent.width != 0 and framebuf_extent.height != 0);
    assert(self.in_frame);
    return Backend.acquireNextImage(self, framebuf_extent);
}

pub const DescPool = enum(u64) {
    _,

    pub const Options = struct {
        pub const Cmd = struct {
            name: DebugName,
            layout: DescSetLayout,
            layout_options: *const CombinedPipelineLayout.Options,
            result: *DescSet,
        };
        name: DebugName,
        cmds: []const Cmd,
    };

    pub fn init(gx: *Ctx, options: @This().Options) @This() {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();
        return Backend.descPoolCreate(gx, options);
    }

    pub fn deinit(self: @This(), gx: *Ctx) void {
        return Backend.descPoolDestroy(gx, self);
    }

    pub inline fn fromBackendType(value: Backend.DescPool) @This() {
        comptime assert(@sizeOf(Backend.DescPool) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.DescPool {
        comptime assert(@sizeOf(Backend.DescPool) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }
};

pub const DescSetLayout = enum(u64) {
    _,

    pub inline fn fromBackendType(value: Backend.DescSetLayout) @This() {
        comptime assert(@sizeOf(Backend.DescSetLayout) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.DescSetLayout {
        comptime assert(@sizeOf(Backend.DescSetLayout) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }
};

pub const DescSet = enum(u64) {
    _,

    pub inline fn fromBackendType(value: Backend.DescSet) @This() {
        comptime assert(@sizeOf(Backend.DescSet) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.DescSet {
        comptime assert(@sizeOf(Backend.DescSet) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }
};

// https://registry.khronos.org/vulkan/specs/1.3/html/chap33.html#limits-minmax
pub const StorageBufSize = u27;

pub const BufKind = packed struct {
    transfer_src: bool = false,
    transfer_dst: bool = false,
    uniform_texel: bool = false,
    storage_texel: bool = false,
    uniform: bool = false,
    storage: bool = false,
    index: bool = false,
    vertex: bool = false,
    indirect: bool = false,
    shader_device_address: bool = false,

    inline fn checkCast(comptime self: @This(), comptime rtype: @This()) void {
        if (comptime !containsBits(self, rtype)) {
            @compileError(std.fmt.comptimePrint("cannot cast {} to {}", .{ self, rtype }));
        }
    }

    inline fn assertNonZero(self: @This()) void {
        const Int = std.meta.Int(.unsigned, @bitSizeOf(@TypeOf(self)));
        const int: Int = @bitCast(self);
        assert(int != 0);
    }

    inline fn assertZero(self: @This()) void {
        const Int = std.meta.Int(.unsigned, @bitSizeOf(@TypeOf(self)));
        const int: Int = @bitCast(self);
        assert(int == 0);
    }
};

pub fn Buf(kind: BufKind) type {
    return enum(u64) {
        const Self = @This();

        pub const View = struct {
            buf: Buf(kind),
            offset: u64,
            size: u64,

            pub inline fn as(
                self: @This(),
                comptime result_kind: BufKind,
            ) Buf(result_kind).View {
                return .{
                    .buf = self.buf.as(result_kind),
                    .offset = self.offset,
                    .size = self.size,
                };
            }

            pub fn unsized(self: @This()) UnsizedView {
                return .{
                    .buf = self.buf,
                    .offset = self.offset,
                };
            }
        };

        pub const UnsizedView = struct {
            buf: Buf(kind),
            offset: u64,

            pub fn as(
                self: @This(),
                comptime result_kind: BufKind,
            ) UnsizedView(result_kind) {
                return .{
                    .buf = self.buf,
                    .offset = self.offset,
                };
            }

            pub fn sized(self: @This(), size: u64) View {
                return .{
                    .buf = self.buf,
                    .offset = self.offset,
                    .size = size,
                };
            }
        };

        pub const Options = struct {
            name: DebugName,
            size: u64,
        };

        pub inline fn as(self: Self, comptime result_kind: BufKind) Buf(result_kind) {
            kind.checkCast(result_kind);
            return @enumFromInt(@intFromEnum(self));
        }

        pub inline fn fromBackendType(value: Backend.Buf) @This() {
            comptime assert(@sizeOf(Backend.Buf) == @sizeOf(@This()));
            return @enumFromInt(@intFromEnum(value));
        }

        pub inline fn asBackendType(self: @This()) Backend.Buf {
            comptime assert(@sizeOf(Backend.Buf) == @sizeOf(@This()));
            return @enumFromInt(@intFromEnum(self));
        }

        _,
    };
}

pub const DescUpdateCmd = struct {
    pub const Value = union(enum) {
        pub const Tag: type = @typeInfo(@This()).@"union".tag_type.?;
        pub const CombinedImageSampler = struct {
            pub const Layout = enum {
                read_only,
                attachment,
            };

            view: ImageView,
            sampler: Sampler,
            layout: Layout,
        };

        storage_buf: Buf(.{ .storage = true }).View,
        uniform_buf: Buf(.{ .uniform = true }).View,
        combined_image_sampler: CombinedImageSampler,
    };

    set: DescSet,
    binding: u32,
    index: u8 = 0,
    value: Value,
};

/// Submit descriptor set update commands. Fastest when sorted. Copy commands not currently
/// supported.
pub fn updateDescSets(self: *@This(), cmds: []const DescUpdateCmd) void {
    Backend.descSetsUpdate(self, cmds);
}

/// Will blocks until the next frame in flight's resources can be reclaimed.
pub fn beginFrame(self: *@This()) void {
    const zone = CpuZone.begin(.{
        .src = @src(),
        .color = global_options.blocking_zone_color,
    });
    defer zone.end();
    assert(!self.in_frame);
    self.in_frame = true;
    Backend.beginFrame(self);
    self.tracy_queries[self.frame] = 0;
}

pub const ImageHandle = enum(u64) {
    _,

    pub inline fn fromBackendType(value: Backend.Image) @This() {
        comptime assert(@sizeOf(Backend.Image) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.Image {
        comptime assert(@sizeOf(Backend.Image) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }
};

pub const ComponentMapping = struct {
    pub const Swizzle = enum {
        identity,
        zero,
        one,
        r,
        g,
        b,
        a,
    };

    r: Swizzle = .identity,
    g: Swizzle = .identity,
    b: Swizzle = .identity,
    a: Swizzle = .identity,
};

// XXX: rename image init options to options, and options to descriptor?
/// Represents a GPU image. The type is generic over color vs depth/stencil as this affects the
/// allocation requirements, after allocation is complete there is no downside to converting to
/// `Image(.any)` if desired.
pub fn Image(kind: ImageKind) type {
    return struct {
        /// The backend's image handle.
        handle: ImageHandle,
        /// The image view. On backends that don't differentiate between views and handles, this is
        /// just a duplicate of the image handle.
        view: ImageView,
        // XXX: make actual handle nullable if gonna store this way?
        /// The memory for this image if it has a dedicated allocation.
        dedicated_memory: ?MemoryUnsized,

        pub const AllocOptions = union(enum) {
            /// Let the driver decide whether to bump allocate the image into the given buffer,
            /// updating offset to reflect the allocation, or to create a dedicated allocation. Use
            /// this approach unless you have a really good reason not to.
            auto: struct {
                memory: Memory(memory_kind),
                offset: *u64,
            },
            /// Creates a dedicated allocation for this image.
            dedicated: void,
            /// Place the image beginning at the start of the given memory view. The caller is
            /// responsible for ensuring proper alignment, not overrunning the buffer, and checking
            /// that this image does not require a dedicated allocation on the current hardware.
            place: struct {
                memory: Memory(memory_kind),
                offset: u64,
            },

            const memory_kind: MemoryKind = switch (kind) {
                .color => .color_image,
                .depth_stencil => |format| .{ .depth_stencil_image = format },
                .any => .any,
            };

            fn asAny(self: @This()) Image(.any).AllocOptions {
                return switch (self) {
                    .auto => |auto| .{ .auto = .{
                        .memory = auto.memory.as(.any),
                        .offset = auto.offset,
                    } },
                    .dedicated => .dedicated,
                    .place => |place| .{ .place = .{
                        .memory = place.memory.as(.any),
                        .offset = place.offset,
                    } },
                };
            }
        };

        fn backendOptions(options: @This().Options) btypes.ImageOptions {
            assert(options.extent.width * options.extent.height * options.extent.depth > 0);
            assert(options.mip_levels > 0);
            assert(options.array_layers > 0);
            return .{
                .flags = options.flags,
                .dimensions = options.dimensions,
                .format = switch (kind) {
                    .depth_stencil => |format| format,
                    .color => options.format,
                    .any => unreachable,
                },
                .extent = options.extent,
                .mip_levels = options.mip_levels,
                .array_layers = options.array_layers,
                .samples = options.samples,
                .usage = switch (kind) {
                    .depth_stencil => .{
                        .transfer_src = options.usage.transfer_src,
                        .transfer_dst = options.usage.transfer_dst,
                        .sampled = options.usage.sampled,
                        .storage = options.usage.storage,
                        .color_attachment = false,
                        .depth_stencil_attachment = options.usage.depth_stencil_attachment,
                        .input_attachment = options.usage.input_attachment,
                    },
                    .color => .{
                        .transfer_src = options.usage.transfer_src,
                        .transfer_dst = options.usage.transfer_dst,
                        .sampled = options.usage.sampled,
                        .storage = options.usage.storage,
                        .color_attachment = options.usage.color_attachment,
                        .depth_stencil_attachment = false,
                        .input_attachment = options.usage.input_attachment,
                    },
                    .any => unreachable,
                },
                .components = options.components,
                .aspect = options.aspect,
                .base_mip_level = options.base_mip_level,
                .mip_level_count = options.mip_level_count,
                .base_array_layer = options.base_array_layer,
                .array_layer_count = options.array_layer_count,
            };
        }

        pub const ColorOptions = struct {
            const Usage = packed struct {
                transfer_src: bool = false,
                transfer_dst: bool = false,
                sampled: bool = false,
                storage: bool = false,
                color_attachment: bool = false,
                input_attachment: bool = false,
            };

            flags: ImageFlags,
            dimensions: Dimensions,
            format: ImageFormat,
            extent: ImageExtent,
            mip_levels: u16,
            array_layers: u16,
            samples: Samples,
            usage: Usage,
            components: ComponentMapping = .{},
            aspect: ImageAspect,
            // XXX: defaults, same in depth stencil
            base_mip_level: u32 = 0,
            mip_level_count: u32 = 1,
            base_array_layer: u32 = 0,
            array_layer_count: u32 = 1,

            pub fn memoryRequirements(self: @This(), gx: *Ctx) MemoryRequirements {
                return Backend.imageMemoryRequirements(gx, backendOptions(self));
            }
        };

        pub const DepthStencilOptions = struct {
            const Usage = packed struct {
                transfer_src: bool = false,
                transfer_dst: bool = false,
                sampled: bool = false,
                storage: bool = false,
                depth_stencil_attachment: bool = false,
                input_attachment: bool = false,
            };

            pub fn memoryRequirements(self: @This(), gx: *Ctx) MemoryRequirements {
                return Backend.imageMemoryRequirements(gx, backendOptions(self));
            }

            flags: ImageFlags,
            dimensions: Dimensions,
            extent: ImageExtent,
            mip_levels: u16,
            array_layers: u16,
            samples: Samples,
            usage: Usage = .{},
            components: ComponentMapping = .{},
            aspect: ImageAspect,
            base_mip_level: u32 = 0,
            mip_level_count: u32 = 1,
            base_array_layer: u32 = 0,
            array_layer_count: u32 = 1,
        };

        pub const InitOptions = struct {
            name: DebugName,
            alloc: Image(kind).AllocOptions,
            image: Image(kind).Options,
        };

        pub const Options = switch (kind) {
            .color => ColorOptions,
            .depth_stencil => DepthStencilOptions,
            .any => unreachable,
        };

        pub fn init(gx: *Ctx, options: InitOptions) @This() {
            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();
            const result = Backend.imageCreate(
                gx,
                options.name,
                options.alloc.asAny(),
                Image(kind).backendOptions(options.image),
            );
            if (result.dedicated_memory) |dedicated_memory| {
                tracy.alloc(.{
                    .ptr = @ptrFromInt(@intFromEnum(dedicated_memory.unsized)),
                    .size = dedicated_memory.size,
                    .pool_name = Ctx.tracy_gpu_pool,
                });
                return .{
                    .handle = result.handle,
                    .view = result.view,
                    .dedicated_memory = dedicated_memory.unsized,
                };
            } else {
                return .{
                    .handle = result.handle,
                    .view = result.view,
                    .dedicated_memory = null,
                };
            }
        }

        pub fn deinit(self: @This(), gx: *Ctx) void {
            Backend.imageDestroy(gx, self.asAny());
        }

        pub fn asAny(self: @This()) Image(.any) {
            return .{
                .handle = @enumFromInt(@intFromEnum(self.handle)),
                .view = self.view,
                .dedicated_memory = self.dedicated_memory,
            };
        }
    };
}

pub const ImageView = enum(u64) {
    _,

    pub const Sized2D = struct {
        view: ImageView,
        extent: Extent2D,
    };

    pub inline fn fromBackendType(value: Backend.ImageView) @This() {
        comptime assert(@sizeOf(Backend.ImageView) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.ImageView {
        comptime assert(@sizeOf(Backend.ImageView) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }
};

pub const ImageKind = union(enum) {
    color,
    depth_stencil: ImageFormat,
    any,

    fn eq(lhs: @This(), rhs: @This()) bool {
        return switch (lhs) {
            .color => rhs == .color,
            .depth_stencil => |lhs_format| switch (rhs) {
                .depth_stencil => |rhs_format| lhs_format == rhs_format,
                else => false,
            },
            .any => rhs == .any,
        };
    }

    // XXX: needed?
    fn castAllowed(self: @This(), as: @This()) bool {
        return as == .any or self.eq(as);
    }
};

pub const ImageFormat = enum(i32) {
    undefined = Backend.named_image_formats.undefined,
    r8g8b8a8_srgb = Backend.named_image_formats.r8g8b8a8_srgb,
    d24_unorm_s8_uint = Backend.named_image_formats.d24_unorm_s8_uint,

    _,

    pub inline fn fromBackendType(value: Backend.ImageFormat) @This() {
        comptime assert(@sizeOf(Backend.ImageFormat) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.ImageFormat {
        comptime assert(@sizeOf(Backend.ImageFormat) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }

    pub fn asBackendSlice(self: []const @This()) []const Backend.ImageFormat {
        comptime assert(@sizeOf(@This()) == @sizeOf(Backend.ImageFormat));
        comptime assert(@alignOf(@This()) == @alignOf(Backend.ImageFormat));
        return @ptrCast(self);
    }
};

// XXX: naming? how do you create the other layers if doing an array?
pub const Dimensions = enum {
    @"1d",
    @"2d",
    @"3d",
    cube,
    @"1d_array",
    @"2d_array",
    cube_array,
};

pub const Samples = enum {
    @"1",
    @"2",
    @"4",
    @"8",
    @"16",
    @"32",
    @"64",
};

pub const ImageFlags = packed struct {
    cube_compatible: bool = false,
    @"2d_array_compatible": bool = false,
};

pub const ImageAspect = packed struct {
    color: bool = false,
    depth: bool = false,
    stencil: bool = false,

    fn assertNonZero(self: @This()) void {
        const Int = std.meta.Int(.unsigned, @bitSizeOf(@TypeOf(self)));
        const int: Int = @bitCast(self);
        assert(int != 0);
    }
};

pub const ImageExtent = struct {
    width: u32,
    height: u32,
    depth: u32,
};

pub const MemoryUnsized = enum(u64) {
    _,

    pub fn deinit(self: @This(), gx: *Ctx) void {
        tracy.free(.{
            .ptr = @ptrFromInt(@intFromEnum(self)),
            .pool_name = tracy_gpu_pool,
        });
        Backend.memoryDestroy(gx, self);
    }

    pub inline fn fromBackendType(value: Backend.Memory) @This() {
        comptime assert(@sizeOf(Backend.Memory) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.Memory {
        comptime assert(@sizeOf(Backend.Memory) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }
};

pub fn Memory(k: MemoryKind) type {
    return struct {
        unsized: MemoryUnsized,
        size: u64,

        pub const kind = k;

        pub const InitNoFormatOptions = struct {
            name: DebugName,
            size: u64,
        };

        pub const InitDepthStencilFormatOptions = struct {
            name: DebugName,
            size: u64,
            format: ImageFormat,
        };

        pub const Options = switch (kind) {
            .buf => InitNoFormatOptions,
            .color_image => InitNoFormatOptions,
            .depth_stencil_image => InitDepthStencilFormatOptions,
            .any => unreachable,
        };

        pub inline fn init(gx: *Ctx, options: @This().Options) @This() {
            comptime switch (kind) {
                .buf => |buffer_kind| buffer_kind.assertNonZero(),
                .color_image, .depth_stencil_image => {},
                .any => unreachable,
            };

            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();

            assert(options.size > 0);

            // XXX: explicit ascription
            const any = Backend.memoryCreate(gx, .{
                .name = options.name,
                .usage = switch (kind) {
                    .buf => |buf| .{ .buf = buf },
                    .color_image => .color_image,
                    .depth_stencil_image => |format| .{ .depth_stencil_image = .{
                        .format = format,
                    } },
                    .any => unreachable,
                },
                .access = .none,
                .size = options.size,
            });
            tracy.alloc(.{
                .ptr = @ptrFromInt(@intFromEnum(any)),
                .size = options.size,
                .pool_name = tracy_gpu_pool,
            });
            return .{
                .unsized = @enumFromInt(@intFromEnum(any)),
                .size = options.size,
            };
        }

        pub fn deinit(self: @This(), gx: *Ctx) void {
            self.unsized.deinit(gx);
        }

        pub inline fn as(
            self: @This(),
            comptime result_kind: MemoryKind,
        ) Memory(result_kind) {
            MemoryKind.checkCast(kind, result_kind);
            return .{
                .unsized = self.unsized,
                .size = self.size,
            };
        }
    };
}

pub const MemoryKind = union(enum) {
    color_image: void,
    depth_stencil_image: ImageFormat,
    buf: BufKind,
    any: void,

    fn checkCast(comptime self: ?@This(), comptime rtype: @This()) void {
        comptime b: {
            if (rtype == .any) break :b;
            if (self) |ltype| {
                if (std.meta.activeTag(ltype) == std.meta.activeTag(rtype)) {
                    switch (ltype) {
                        .image => |image| break :b assert(image.castAllowed(rtype.image)),
                        .buf => |buf| break :b buf.checkCast(rtype.buf),
                    }
                }
            }
            const self_name = if (self) |s| @tagName(s) else "null";
            @compileError("cannot cast " ++ self_name ++ " to " ++ @tagName(rtype));
        }
    }
};

pub fn MemoryView(kind: MemoryKind) type {
    return struct {
        memory: MemoryUnsized(kind),
        offset: u64,
        size: u64,

        pub inline fn as(
            self: @This(),
            comptime result_kind: MemoryKind,
        ) MemoryView(result_kind) {
            kind.checkCast(result_kind);
            return .{
                .memory = @enumFromInt(@intFromEnum(self.memory)),
                .offset = self.offset,
                .size = self.size,
            };
        }
    };
}

pub fn MemoryViewUnsized(kind: MemoryKind) type {
    return struct {
        memory: MemoryUnsized(kind),
        offset: u64,

        pub inline fn as(
            self: @This(),
            comptime result_kind: MemoryKind,
        ) MemoryViewUnsized(result_kind) {
            MemoryKind.checkCast(kind, result_kind);
            return .{
                .memory = self.memory.as(result_kind),
                .offset = self.offset,
            };
        }
    };
}

// XXX: naming
/// Some APIs have separate handles for descriptor set layouts and pipelines (e.g. Vulkan), others
/// have a single handle that refers to both the pipeline layout state and the descriptor set layout
/// state (e.g. DirectX 12). On APIs that don't distinguish between pipeline and descriptor set
/// layouts, both handles are equivalent.
pub const CombinedPipelineLayout = struct {
    pipeline: PipelineLayout,
    desc_set: DescSetLayout,

    pub const Options = struct {
        pub const Desc = struct {
            pub const Kind = union(enum) {
                uniform_buffer: struct {
                    // We only support sizes up to 2^14, as Vulkan implementations don't have to support
                    // uniform buffers larger than this:
                    // https://registry.khronos.org/vulkan/specs/1.3/html/chap33.html#limits-minmax
                    size: u14,
                },
                storage_buffer: void,
                combined_image_sampler: void,
            };
            pub const Stages = packed struct(u2) {
                vertex: bool = false,
                fragment: bool = false,
            };

            name: []const u8,
            kind: Kind,
            count: u32 = 1,
            stages: Stages,
        };

        name: DebugName,
        descs: []const Desc,

        pub fn getBindingIndex(comptime self: *const @This(), comptime name: []const u8) u32 {
            const result = comptime for (self.descs, 0..) |desc, i| {
                if (std.mem.eql(u8, desc.name, name)) {
                    break i;
                }
            } else @compileError("no such binding " ++ name);
            return result;
        }
    };

    pub fn init(gx: *Ctx, options: @This().Options) CombinedPipelineLayout {
        return Backend.combinedPipelineLayoutCreate(gx, options);
    }

    pub fn deinit(self: @This(), gx: *Ctx) void {
        Backend.combinedPipelineLayoutDestroy(gx, self);
    }
};

pub const PipelineLayout = enum(u64) {
    _,

    pub inline fn fromBackendType(value: Backend.PipelineLayout) @This() {
        comptime assert(@sizeOf(Backend.PipelineLayout) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.PipelineLayout {
        comptime assert(@sizeOf(Backend.PipelineLayout) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }
};

pub const PipelineHandle = enum(u64) {
    _,

    pub inline fn fromBackendType(value: Backend.Pipeline) @This() {
        comptime assert(@sizeOf(Backend.Pipeline) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.Pipeline {
        comptime assert(@sizeOf(Backend.Pipeline) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }
};

// XXX: move onto pipeline?
pub const InitPipelineCmd = struct {
    pub const Stages = struct {
        pub const max_stages = std.meta.fields(@This()).len;
        vertex: ShaderModule,
        fragment: ShaderModule,
    };
    pub const InputAssembly = union(enum) {
        const Strip = struct { indexed_primitive_restart: bool = false };
        point_list: void,
        line_list: void,
        line_strip: Strip,
        triangle_list: void,
        triangle_strip: Strip,
        line_list_with_adjacency: void,
        line_strip_with_adjacency: Strip,
        triangle_list_with_adjacency: void,
        triangle_strip_with_adjacency: Strip,
        patch_list: void,
    };

    name: DebugName,
    layout: CombinedPipelineLayout,
    stages: Stages,
    result: *Pipeline,
    input_assembly: InputAssembly,
    color_attachment_formats: []const ImageFormat,
    depth_attachment_format: ImageFormat,
    stencil_attachment_format: ImageFormat,
};

pub const ShaderModule = enum(u64) {
    _,

    pub const Options = struct {
        name: DebugName,
        ir: []const u32,
    };

    pub fn init(gx: *Ctx, options: @This().Options) @This() {
        return Backend.shaderModuleCreate(gx, options);
    }

    pub fn deinit(self: @This(), gx: *Ctx) void {
        Backend.shaderModuleDestroy(gx, self);
    }

    pub inline fn fromBackendType(value: Backend.ShaderModule) @This() {
        comptime assert(@sizeOf(Backend.ShaderModule) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.ShaderModule {
        comptime assert(@sizeOf(Backend.ShaderModule) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }
};

pub const Pipeline = struct {
    handle: PipelineHandle,
    layout: PipelineLayout,

    pub fn init(
        gx: *Ctx,
        cmds: []const InitPipelineCmd,
    ) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();
        if (std.debug.runtime_safety) {
            assert(cmds.len < global_options.init_pipelines_buf_len);
            for (cmds) |cmd| {
                // Support for up to 4 is guaranteed by our Vulkan version. Once we update to 1.4,
                // we can bump this to 8.
                assert(cmd.color_attachment_formats.len <= 4);
            }
        }
        Backend.pipelinesCreate(gx, cmds);
    }

    pub fn deinit(self: @This(), gx: *Ctx) void {
        Backend.pipelineDestroy(gx, self);
    }
};

pub const Sampler = enum(u64) {
    _,

    pub const Options = struct {
        pub const Filter = enum {
            nearest,
            linear,
        };
        pub const AddressMode = enum {
            repeat,
            mirrored_repeat,
            clamp_to_edge,
            clamp_to_border,
            mirror_clamp_to_edge,
        };
        pub const AddressModes = struct {
            u: AddressMode,
            v: AddressMode,
            w: AddressMode,

            pub fn initAll(mode: AddressMode) AddressModes {
                return .{
                    .u = mode,
                    .v = mode,
                    .w = mode,
                };
            }
        };
        pub const CompareOp = enum {
            never,
            less,
            equal,
            less_or_equal,
            greater,
            not_equal,
            greater_or_equal,
            always,
        };
        pub const BorderColor = enum {
            float_transparent_black,
            int_transparent_black,
            float_opaque_black,
            int_opaque_black,
            float_opaque_white,
            int_opaque_white,
        };

        mag_filter: Filter,
        min_filter: Filter,
        mipmap_mode: Filter,
        address_mode: AddressModes,
        mip_lod_bias: f32,
        max_anisotropy: enum(u8) {
            none = 0,
            @"1" = 1,
            @"2" = 2,
            @"3" = 3,
            @"4" = 4,
            @"5" = 5,
            @"6" = 6,
            @"7" = 7,
            @"8" = 8,
            @"9" = 9,
            @"10" = 10,
            @"11" = 11,
            @"12" = 12,
            @"13" = 13,
            @"14" = 14,
            @"15" = 15,
            @"16" = 16,
        },
        compare_op: ?CompareOp,
        min_lod: f32,
        max_lod: ?f32,
        border_color: BorderColor,
    };

    pub fn init(gx: *Ctx, name: DebugName, options: @This().Options) Sampler {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();
        return Backend.samplerCreate(gx, name, options);
    }

    pub fn deinit(self: @This(), gx: *Ctx) void {
        Backend.samplerDestroy(gx, self);
    }

    pub inline fn fromBackendType(value: Backend.Sampler) @This() {
        comptime assert(@sizeOf(Backend.Sampler) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.Sampler {
        comptime assert(@sizeOf(Backend.Sampler) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }
};

pub const TimestampCalibration = struct {
    cpu: u64,
    gpu: u64,
    max_deviation: u64,
};

pub fn timestampCalibration(self: *Ctx) TimestampCalibration {
    return Backend.timestampCalibration(self);
}

pub fn waitIdle(self: *const @This()) void {
    Backend.waitIdle(self);
}

fn containsBits(self: anytype, other: @TypeOf(self)) bool {
    const Int = std.meta.Int(.unsigned, @bitSizeOf(@TypeOf(self)));
    const self_bits: Int = @bitCast(self);
    const other_bits: Int = @bitCast(other);
    return self_bits & other_bits == other_bits;
}

fn AsBackendSlice(Item: type) type {
    const BackendItem = @FieldType(Item, "backend");
    comptime assert(@sizeOf(Item) == @sizeOf(BackendItem));
    comptime assert(@alignOf(Item) == @alignOf(BackendItem));
    return struct {
        pub fn mixin(slice: []const Item) []const BackendItem {
            return @ptrCast(slice);
        }
    };
}
