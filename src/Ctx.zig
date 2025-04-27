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
const global_options = @import("root.zig").options;
const builtin = @import("builtin");
const IBackend = @import("ibackend.zig").IBackend;

pub const Backend = IBackend(global_options.Backend).create();
const Ctx = @This();

const tracy_gpu_pool = "gpu";

pub const FramebufSize = struct { u32, u32 };

pub const MemReqs = struct {
    size: u64,
    alignment: u64,
};

pub const DebugName = struct {
    str: [*:0]const u8,
    index: ?usize = null,
};

pub const Semaphore = enum(u64) {
    _,

    pub inline fn fromBackendType(value: Backend.Semaphore) @This() {
        comptime assert(@sizeOf(Backend.Semaphore) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.Semaphore {
        comptime assert(@sizeOf(Backend.Semaphore) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }

    pub fn waitAll(gx: *Ctx, semaphores: []const Semaphore) void {
        var wait_values_buf: [global_options.max_cbs_per_frame]u64 = undefined;
        const wait_values = wait_values_buf[0..semaphores.len];
        for (wait_values) |*value| {
            value.* = gx.frame + gx.frames_in_flight;
        }
        Backend.semaphoresWait(gx, semaphores, wait_values);
    }
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
};

backend: global_options.Backend,

device: Device,

/// May not be changed after initialization.
frames_in_flight: u8,
frame: u64 = 0,
framebuf_size: FramebufSize = .{ 0.0, 0.0 },

combined_pipeline_layout_typed: [global_options.combined_pipeline_layouts.len]CombinedPipelineLayout,

cb_bindings: [global_options.max_frames_in_flight]std.BoundedArray(CmdBufBindings, global_options.max_cbs_per_frame) = @splat(.{}),
cb_semaphores: [global_options.max_frames_in_flight]std.BoundedArray(Semaphore, global_options.max_cbs_per_frame) = @splat(.{}),

max_alignment: bool,

timestamp_queries: bool,

pub fn InitOptionsImpl(BackendInitOptions: type) type {
    return struct {
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
        frames_in_flight: u8,
        framebuf_size: struct { u32, u32 },
        backend: BackendInitOptions,
        device_type_ranks: std.EnumArray(Device.Kind, u8) = default_device_type_ranks,
        timestamp_queries: bool,
        validation: bool,
        // When true, forces storage and uniform buffer alignment to max values. Can be used to
        // debug alignment related errors, or to force the equivalent conservative alignment on all
        // platforms.
        max_alignment: bool = false,
    };
}

pub const InitOptions = InitOptionsImpl(Backend.InitOptions);

pub fn init(options: InitOptions) @This() {
    const zone = tracy.Zone.begin(.{ .name = "gpu init", .src = @src() });
    defer zone.end();
    log.debug("Initializing GPU frontend", .{});

    assert(options.frames_in_flight > 0);
    assert(options.frames_in_flight <= global_options.max_frames_in_flight);

    const backend = Backend.init(options);

    var gx: @This() = .{
        .backend = backend,

        .device = backend.getDevice(),

        .frames_in_flight = options.frames_in_flight,

        .combined_pipeline_layout_typed = undefined,

        .max_alignment = options.max_alignment,

        .timestamp_queries = options.timestamp_queries,
    };

    {
        const semaphore_zone = tracy.Zone.begin(.{ .name = "create semaphores", .src = @src() });
        defer semaphore_zone.end();

        for (&gx.cb_semaphores) |*semaphores| {
            for (&semaphores.buffer) |*semaphore| {
                semaphore.* = Backend.semaphoreCreate(&gx, 0);
            }
        }
    }

    if (gx.max_alignment) {
        gx.device.uniform_buf_offset_alignment = Device.max_uniform_buf_offset_alignment;
        gx.device.storage_buf_offset_alignment = Device.max_storage_buf_offset_alignment;
    }

    comptime var max_descriptors = 0;
    inline for (global_options.combined_pipeline_layouts) |layout| {
        max_descriptors = @max(layout.descriptors.len, max_descriptors);
    }

    inline for (global_options.combined_pipeline_layouts, 0..) |create_options, i| {
        gx.combined_pipeline_layout_typed[i] = CombinedPipelineLayout.init(
            &gx,
            max_descriptors,
            create_options.*,
        );
    }

    return gx;
}

/// Destroys the context. Must not be in use, see `waitIdle`.
pub fn deinit(self: *@This(), gpa: Allocator) void {
    for (self.combined_pipeline_layout_typed) |layout| {
        layout.deinit(self);
    }

    for (&self.cb_semaphores) |semaphores| {
        for (semaphores.buffer) |semaphore| {
            Backend.semaphoreDestroy(self, semaphore);
        }
    }

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
        memory: Memory(.{ .usage = .{ .buf = kind } }),
        buf: Buf(kind),

        pub const InitOptions = struct {
            name: DebugName,
            size: u64,
        };

        pub inline fn init(
            gx: *Ctx,
            options: @This().InitOptions,
        ) DedicatedBuf(kind) {
            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();
            comptime kind.assertNonZero();
            assert(options.size > 0); // Vulkan doesn't support zero sized buffers
            const untyped = Backend.dedicatedBufCreate(gx, options.name, kind, options.size);
            return .{
                .memory = @enumFromInt(@intFromEnum(untyped.memory)),
                .buf = @enumFromInt(@intFromEnum(untyped.buf)),
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
        memory: Memory(.{ .usage = .{ .buf = kind }, .access = .read }),
        buf: Buf(kind),
        data: []const u8,

        pub const InitOptions = struct {
            name: DebugName,
            size: u64,
        };

        pub inline fn init(
            gx: *Ctx,
            options: @This().InitOptions,
        ) @This() {
            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();
            comptime kind.assertNonZero();
            const untyped = Backend.dedicatedReadbackBufCreate(gx, options.name, kind, options.size);
            return .{
                .memory = @enumFromInt(@intFromEnum(untyped.memory)),
                .buf = @enumFromInt(@intFromEnum(untyped.buf)),
                .data = untyped.data,
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
        memory: Memory(.{ .usage = .{ .buf = kind }, .access = .write }),
        buf: Buf(kind),
        data: []volatile anyopaque,

        pub const InitOptions = struct {
            name: DebugName,
            size: u64,
            prefer_device_local: bool,
        };

        pub inline fn init(
            gx: *Ctx,
            options: @This().InitOptions,
        ) DedicatedUploadBuf(kind) {
            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();
            comptime kind.assertNonZero();
            const untyped = Backend.dedicatedUploadBufCreate(
                gx,
                options.name,
                kind,
                options.size,
                options.prefer_device_local,
            );
            return .{
                .memory = @enumFromInt(@intFromEnum(untyped.memory)),
                .buf = @enumFromInt(@intFromEnum(untyped.buf)),
                .data = untyped.data,
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

pub const Wait = struct {
    pub const Stages = packed struct {
        top_of_pipe: bool,
        draw_indirect: bool,
        vertex_input: bool,
        vertex_shader: bool,
        tessellation_control_shader: bool,
        tessellation_evaluation_shader: bool,
        geometry_shader: bool,
        fragment_shader: bool,
        early_fragment_tests: bool,
        late_fragment_tests: bool,
        color_attachment_output: bool,
        compute_shader: bool,
        transfer: bool,
        bottom_of_pipe: bool,
        host: bool,
        all_graphics: bool,
        all_commands: bool,
    };
    semaphore: Semaphore,
    stages: Stages,
};

pub const CmdBufBindings = struct {
    pipeline: ?Pipeline = null,
    indices: ?Buf(.{ .index = true }) = null,
    descriptor_set: ?DescSet(null) = null,
    dynamic_state: bool = false,
};

pub const CombinedCmdBufCreateOptions = struct {
    bindings: *Ctx.CmdBufBindings,
    signal: Ctx.Semaphore,
    kind: Ctx.CmdBufKind,
    loc: *const tracy.SourceLocation,
};

pub fn CombinedCmdBuf(kind: ?CmdBufKind) type {
    return struct {
        cmds: Cmds,
        signal: Semaphore,
        zone: Zone,

        pub inline fn init(gx: *Ctx, loc: *const tracy.SourceLocation) @This() {
            comptime assert(kind != null);

            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();

            const bindings = gx.cb_bindings[gx.frameInFlight()].addOne() catch @panic("OOB");
            bindings.* = .{};
            const signal = gx.cb_semaphores[gx.frameInFlight()].addOneAssumeCapacity().*;

            const untyped = Backend.combinedCmdBufCreate(gx, .{
                .bindings = bindings,
                .signal = signal,
                .kind = kind.?,
                .loc = loc,
            });

            return .{
                .cmds = .{
                    .buf = untyped.cmds.buf,
                    .bindings = untyped.cmds.bindings,
                },
                .signal = untyped.signal,
                .zone = untyped.zone,
            };
        }

        pub inline fn submit(
            self: @This(),
            gx: *Ctx,
            wait: []const Wait,
        ) void {
            comptime assert(kind != null);

            const zone = CpuZone.begin(.{ .src = @src() });
            defer zone.end();

            // Not <= as you can't wait on the final submission!
            assert(wait.len < global_options.max_cbs_per_frame);

            Backend.combinedCmdBufSubmit(gx, self.asUntyped(), kind.?, wait);
        }

        inline fn asUntyped(self: @This()) CombinedCmdBuf(null) {
            return .{
                .cmds = self.cmds,
                .signal = self.signal,
                .zone = self.zone,
            };
        }
    };
}

pub const Cmds = struct {
    buf: CmdBuf,
    /// Cached bindings for de-duplicating commands. If you write to the command buffer
    /// externally, you should update this to reflect your changes. You can set fields to their
    /// defaults to indicate that the state is unknown.
    bindings: *CmdBufBindings,

    pub const AppendGraphicsCmdsOptions = struct {
        cmds: []const DrawCmd,
        loc: *const tracy.SourceLocation,
    };

    pub fn appendGraphicsCmds(
        self: @This(),
        gx: *Ctx,
        options: AppendGraphicsCmdsOptions,
    ) void {
        const zone = CpuZone.begin(.{ .src = @src() });
        defer zone.end();
        if (std.debug.runtime_safety) {
            for (options.cmds) |draw_call| {
                assert(draw_call.args.offset % 4 == 0);
            }
        }
        Backend.cmdBufGraphicsAppend(gx, self, options);
    }

    pub const AppendTransferCmdsOptions = struct {
        cmds: []const TransferCmd,
        loc: *const tracy.SourceLocation,
    };

    pub fn appendTransferCmds(
        self: @This(),
        gx: *Ctx,
        comptime max_regions: u32,
        options: AppendTransferCmdsOptions,
    ) void {
        if (std.debug.runtime_safety) {
            for (options.cmds) |cmd| switch (cmd) {
                .copy_buffer_to_color_image => |cmd_options| {
                    assert(cmd_options.regions.len > 0);
                    assert(cmd_options.regions.len <= max_regions);
                    for (cmd_options.regions) |region| {
                        assert(region.buffer_row_length != 0);
                        assert(region.buffer_image_height != 0);
                        assert(region.layer_count > 0);
                    }
                },
                .copy_buffer_to_buffer => |cmd_options| {
                    assert(cmd_options.regions.len > 0);
                    assert(cmd_options.regions.len <= max_regions);
                    for (cmd_options.regions) |region| {
                        assert(region.size > 0);
                    }
                },
            };
        }

        Backend.cmdBufTransferAppend(gx, self, max_regions, options);
    }
};

pub const CmdBuf = enum(u64) {
    _,

    pub inline fn fromBackendType(value: Backend.CmdBuf) @This() {
        comptime assert(@sizeOf(Backend.CmdBuf) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.CmdBuf {
        comptime assert(@sizeOf(Backend.CmdBuf) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }
};

pub const CmdBufKind = enum {
    present,
    graphics,
};

/// Profiling zones are created automatically when creating or appending to a command buffer. It may
/// be desirable to create them manually when modifying a command buffer from outside the library.
pub const Zone = struct {
    index: u15,

    pub fn beginId(self: @This()) u16 {
        return @as(u16, self.index) * 2;
    }

    pub fn endId(self: @This()) u16 {
        return @as(u16, self.index) * 2 + 1;
    }

    pub const BeginOptions = struct {
        command_buffer: CmdBuf,
        tracy_queue: TracyQueue,
        loc: *const tracy.SourceLocation,
    };

    pub fn begin(gx: *Ctx, options: BeginOptions) Zone {
        return Backend.zoneBegin(gx, options);
    }

    pub const EndOptions = struct {
        command_buffer: CmdBuf,
        tracy_queue: TracyQueue,
    };

    pub fn end(self: @This(), gx: *Ctx, options: EndOptions) void {
        Backend.zoneEnd(gx, self, options);
    }
};

pub const DrawCmd = struct {
    pub const IndexedIndirect = extern struct {
        pub const Index = u16;

        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    };

    pub const Indirect = extern struct {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    };

    combined_pipeline: CombinedPipeline(null),
    args: Buf(.{ .indirect = true }).UnsizedView,
    args_count: u32,
    indices: ?Buf(.{ .index = true }) = null,
    descriptor_set: DescSet(null),
};

/// Presents the rendered image. Some drivers under some circumstances will block here  waiting for
/// the swapchain image instead of when acquiring it. Returns the number of nanoseconds spent
/// blocking.
pub fn present(self: *@This()) u64 {
    const zone = CpuZone.begin(.{ .src = @src() });
    defer zone.end();
    return Backend.present(self);
}

/// Acquires the next swapchain image, blocking until it's available if necessary. Returns the
/// nanoseconds spent blocking.
///
/// Returns null if the swapchain needed to be recreated, in which case you should drop this frame.
pub fn acquireNextImage(self: *@This(), framebuf_size: Ctx.FramebufSize) ?u64 {
    const zone = CpuZone.begin(.{ .src = @src() });
    defer zone.end();

    assert(framebuf_size[0] != 0 and framebuf_size[0] != 0);
    self.framebuf_size = framebuf_size;
    return Backend.acquireNextImage(self);
}

pub const DescPool = enum(u64) {
    _,

    pub const InitOptions = struct {
        pub const Cmd = struct {
            name: DebugName,
            layout: DescSetLayout,
            layout_create_options: *const CombinedPipelineLayout.InitOptions,
            result: *DescSet(null),
        };
        name: DebugName,
        cmds: []const Cmd,
    };

    pub fn init(
        gx: *Ctx,
        comptime max_cmds: u32,
        options: @This().InitOptions,
    ) @This() {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();
        assert(options.cmds.len <= max_cmds);
        return Backend.descriptorPoolCreate(gx, max_cmds, options);
    }

    pub fn deinit(self: @This(), gx: *Ctx) void {
        return Backend.descriptorPoolDestroy(gx, self);
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

pub fn DescSet(layout: ?*const CombinedPipelineLayout.InitOptions) type {
    return enum(u64) {
        const type_dependency = layout;

        _,

        pub inline fn asUntyped(self: @This()) DescSet(null) {
            return @enumFromInt(@intFromEnum(self));
        }

        pub inline fn fromBackendType(value: Backend.DescSet) @This() {
            comptime assert(@sizeOf(Backend.DescSet) == @sizeOf(@This()));
            return @enumFromInt(@intFromEnum(value));
        }

        pub inline fn asBackendType(self: @This()) Backend.DescSet {
            comptime assert(@sizeOf(Backend.DescSet) == @sizeOf(@This()));
            return @enumFromInt(@intFromEnum(self));
        }
    };
}

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

        pub const InitOptions = struct {
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
            view: ImageView,
            sampler: Sampler,
            layout: ImageOptions.Layout,
        };

        storage_buffer_view: Buf(.{}).View,
        uniform_buffer_view: Buf(.{}).View,
        combined_image_sampler: CombinedImageSampler,
    };

    set: DescSet(null),
    binding: u32,
    index: u8 = 0,
    value: Value,
};

/// Submit descriptor set update commands. Fastest when sorted. Copy commands not currently
/// supported.
pub fn updateDescSets(
    self: *@This(),
    comptime max_updates: u32,
    cmds: []const DescUpdateCmd,
) void {
    if (cmds.len == 0) return;
    assert(cmds.len <= max_updates);
    Backend.descriptorSetsUpdate(self, max_updates, cmds);
}

/// Returns the current frame in flight. May be used to index resources that are buffered per frame.
pub fn frameInFlight(self: *const @This()) u8 {
    return @intCast(self.frame % self.frames_in_flight);
}

/// Starts a new frame. If this frame in flight's command pool is still in use, blocks until it is
/// available. Returns the time spent blocking in nanoseconds.
pub fn frameStart(self: *@This()) ?u64 {
    const zone = CpuZone.begin(.{ .src = @src() });
    defer zone.end();

    self.frame += 1;

    const block_ns = b: {
        const semaphores = self.cb_semaphores[self.frameInFlight()].constSlice();
        var wait_values_buf: [global_options.max_cbs_per_frame]u64 = undefined;
        const wait_values = wait_values_buf[0..semaphores.len];
        for (wait_values) |*value| {
            value.* = self.frame;
        }

        const wait_zone = CpuZone.begin(.{
            .src = @src(),
            .name = "blocking: command pool",
            .color = global_options.blocking_zone_color,
        });
        defer wait_zone.end();
        var wait_timer = std.time.Timer.start() catch |err| @panic(@errorName(err));
        Backend.semaphoresWait(self, semaphores, wait_values);
        break :b wait_timer.lap();
    };

    self.cb_bindings[self.frameInFlight()].clear();
    self.cb_semaphores[self.frameInFlight()].clear();

    Backend.frameStart(self);

    return block_ns;
}

pub fn Image(kind: ImageKind) type {
    return enum(u64) {
        _,

        pub const InitColorOptions = struct {
            pub const Kind = struct {
                tiling: ImageOptions.Tiling,
                transient_attachment: bool,

                pub fn asKind(self: @This()) ImageKind {
                    return .{
                        .format = .color,
                        .tiling = self.tiling,
                        .transient_attachment = self.transient_attachment,
                    };
                }
            };

            const Usage = packed struct {
                transfer_src: bool = false,
                transfer_dst: bool = false,
                sampled: bool = false,
                storage: bool = false,
                color_attachment: bool = false,
                input_attachment: bool = false,
            };

            name: DebugName,
            flags: ImageOptions.Flags,
            dimensions: ImageOptions.Dimensions,
            format: ImageOptions.Format.Color,
            extent: ImageOptions.Extent,
            mip_levels: u16,
            array_layers: u16,
            samples: ImageOptions.Samples,
            exclusive: bool,
            initial_layout: ImageOptions.Layout,
            usage: Usage,
            location: DeviceMemViewUnsized(.{ .usage = .{ .image = kind } }),
        };

        pub const InitDepthStencilOptions = struct {
            pub const Kind = struct {
                format: ImageOptions.Format.DepthStencil,
                tiling: ImageOptions.Tiling,
                transient_attachment: bool,

                pub fn asKind(self: @This()) ImageKind {
                    return .{
                        .format = .{ .depth_stencil = self.format },
                        .tiling = self.tiling,
                        .transient_attachment = self.transient_attachment,
                    };
                }
            };

            const Usage = packed struct {
                transfer_src: bool = false,
                transfer_dst: bool = false,
                sampled: bool = false,
                storage: bool = false,
                depth_stencil_attachment: bool = false,
                input_attachment: bool = false,
            };

            name: DebugName,
            flags: ImageOptions.Flags,
            dimensions: ImageOptions.Dimensions,
            extent: ImageOptions.Extent,
            mip_levels: u16,
            array_layers: u16,
            samples: ImageOptions.Samples,
            exclusive: bool,
            initial_layout: ImageOptions.Layout,
            usage: Usage = .{},
            location: DeviceMemViewUnsized(.{ .usage = .{ .image = kind } }),
        };

        pub const InitOptions = if (kind.format) |format| switch (format) {
            .color => InitColorOptions,
            .depth_stencil => InitDepthStencilOptions,
        } else @compileError("missing format");

        pub inline fn init(gx: *Ctx, options: @This().InitOptions) @This() {
            comptime assert(kind.nonNull());

            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();

            assert(options.extent.width * options.extent.height * options.extent.depth > 0);
            assert(options.mip_levels > 0);
            assert(options.array_layers > 0);

            const handle = Backend.imageCreate(gx, .{
                .name = options.name,
                .flags = options.flags,
                .tiling = kind.tiling.?,
                .dimensions = options.dimensions,
                .format = switch (kind.format.?) {
                    .depth_stencil => |format| .{ .depth_stencil = format },
                    .color => .{ .color = options.format },
                },
                .extent = options.extent,
                .mip_levels = options.mip_levels,
                .array_layers = options.array_layers,
                .samples = options.samples,
                .exclusive = options.exclusive,
                .initial_layout = options.initial_layout,
                .usage = switch (kind.format.?) {
                    .depth_stencil => .{
                        .transfer_src = options.usage.transfer_src,
                        .transfer_dst = options.usage.transfer_dst,
                        .sampled = options.usage.sampled,
                        .storage = options.usage.storage,
                        .color_attachment = false,
                        .depth_stencil_attachment = options.usage.depth_stencil_attachment,
                        .input_attachment = options.usage.input_attachment,
                        .transient_attachment = kind.transient_attachment.?,
                    },
                    .color => .{
                        .transfer_src = options.usage.transfer_src,
                        .transfer_dst = options.usage.transfer_dst,
                        .sampled = options.usage.sampled,
                        .storage = options.usage.storage,
                        .color_attachment = options.usage.color_attachment,
                        .depth_stencil_attachment = false,
                        .input_attachment = options.usage.input_attachment,
                        .transient_attachment = kind.transient_attachment.?,
                    },
                },
                .location = options.location.as(.{}),
            });
            return @enumFromInt(@intFromEnum(handle));
        }

        pub fn deinit(self: @This(), gx: *Ctx) void {
            Backend.imageDestroy(gx, self.as(.{}));
        }

        pub fn memReqs(self: @This(), gx: *Ctx) MemReqs {
            return Backend.imageMemReqs(gx, self.as(.{}));
        }

        pub inline fn as(self: @This(), comptime result_kind: ImageKind) Image(result_kind) {
            if (!comptime kind.castAllowed(result_kind)) {
                @compileError("cannot convert " ++ @typeName(@This()) ++ " to " ++ @typeName(Image(result_kind)));
            }

            return @enumFromInt(@intFromEnum(self));
        }

        pub inline fn fromBackendType(value: Backend.Image) @This() {
            comptime assert(@sizeOf(Backend.Image) == @sizeOf(@This()));
            return @enumFromInt(@intFromEnum(value));
        }

        pub inline fn asBackendType(self: @This()) Backend.Image {
            comptime assert(@sizeOf(Backend.Image) == @sizeOf(@This()));
            return @enumFromInt(@intFromEnum(self));
        }
    };
}

pub const ImageKind = struct {
    pub const Format = union(enum) {
        color,
        depth_stencil: ImageOptions.Format.DepthStencil,

        fn eq(lhs: @This(), rhs: @This()) bool {
            return switch (lhs) {
                .color => rhs == .color,
                .depth_stencil => |lhs_format| switch (rhs) {
                    .depth_stencil => |rhs_format| lhs_format == rhs_format,
                    else => false,
                },
            };
        }
    };
    format: ?Format = null,
    tiling: ?ImageOptions.Tiling = null,
    transient_attachment: ?bool = null,

    fn nonNull(self: @This()) bool {
        return self.format != null and self.tiling != null and self.transient_attachment != null;
    }

    fn castAllowed(self: ImageKind, as: ImageKind) bool {
        if (as.format != null and self.format == null and !self.format.?.eq(as.format.?)) return false;
        if (as.tiling != null and self.tiling != as.tiling) return false;
        if (as.transient_attachment != null and self.transient_attachment != as.transient_attachment) return false;
        return true;
    }
};

pub const ImageOptions = struct {
    pub const Dimensions = enum {
        @"1d",
        @"2d",
        @"3d",
    };

    pub const Extent = struct {
        width: u32,
        height: u32,
        depth: u32,
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

    pub const Tiling = enum {
        linear,
        optimal,
    };

    pub const Format = union(enum) {
        pub const Color = enum {
            r8g8b8a8_srgb,
        };
        pub const DepthStencil = enum {
            d24_unorm_s8_uint,
        };
        color: Color,
        depth_stencil: DepthStencil,

        fn eq(lhs: @This(), rhs: @This()) bool {
            if (std.meta.activeTag(lhs) != std.meta.activeTag(rhs)) return false;
            if (std.meta.activeTag(lhs) == .depth_stencil) {
                return lhs.depth_stencil == rhs.depth_stencil;
            }
            return true;
        }
    };

    pub const Usage = packed struct {
        transfer_src: bool = false,
        transfer_dst: bool = false,
        sampled: bool = false,
        storage: bool = false,
        color_attachment: bool = false,
        depth_stencil_attachment: bool = false,
        transient_attachment: bool = false,
        input_attachment: bool = false,
    };

    pub const Layout = enum {
        undefined,
        general,
        color_attachment_optimal,
        depth_stencil_attachment_optimal,
        depth_stencil_read_only_optimal,
        shader_read_only_optimal,
        transfer_src_optimal,
        transfer_dst_optimal,
        preinitialized,
        depth_read_only_stencil_attachment_optimal,
        depth_attachment_stencil_read_only_optimal,
        depth_attachment_optimal,
        depth_read_only_optimal,
        stencil_attachment_optimal,
        stencil_read_only_optimal,
        read_only_optimal,
        attachment_optimal,
    };

    pub const Flags = packed struct {
        sparse_binding: bool = false,
        sparse_residency: bool = false,
        sparse_aliased: bool = false,
        mutable_format: bool = false,
        cube_compatible: bool = false,
        alias: bool = false,
        split_instance_bind_regions: bool = false,
        @"2d_array_compatible": bool = false,
        block_texel_view_compatible: bool = false,
        extended_usage: bool = false,
        protected: bool = false,
    };

    name: DebugName,
    flags: Flags,
    tiling: Tiling,
    dimensions: Dimensions,
    format: Format,
    extent: Extent,
    mip_levels: u16,
    array_layers: u16,
    samples: Samples,
    exclusive: bool,
    initial_layout: Layout,
    usage: Usage,
    location: DeviceMemViewUnsized(.{}),
};

pub const ImageView = enum(u64) {
    _,

    pub const InitOptions = struct {
        pub const Kind = enum {
            @"1d",
            @"2d",
            @"3d",
            cube,
            @"1d_array",
            @"2d_array",
            cube_array,
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

        pub const Aspect = packed struct {
            color: bool = false,
            depth: bool = false,
            stencil: bool = false,
            metadata: bool = false,
            plane_0: bool = false,
            plane_1: bool = false,
            plane_2: bool = false,
        };

        name: DebugName,
        image: Image(.{}),
        kind: Kind,
        format: ImageOptions.Format,
        components: ComponentMapping = .{},
        base_mip_level: u32 = 0,
        level_count: u32 = 1,
        base_array_layer: u32 = 0,
        layer_count: u32 = 1,
        aspect: Aspect,
    };

    pub fn init(gx: *Ctx, options: @This().InitOptions) ImageView {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();
        return Backend.imageViewCreate(gx, options);
    }

    pub fn deinit(self: @This(), gx: *Ctx) void {
        Backend.imageViewDestroy(gx, self);
    }

    pub inline fn fromBackendType(value: Backend.ImageView) @This() {
        comptime assert(@sizeOf(Backend.ImageView) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.ImageView {
        comptime assert(@sizeOf(Backend.ImageView) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }
};

pub fn Memory(k: DeviceMemKind) type {
    return enum(u64) {
        const Self = @This();

        pub const kind = k;

        pub const InitNoFormatOptions = struct {
            name: DebugName,
            size: u64,
        };

        pub const InitNoFormatWriteOptions = struct {
            name: DebugName,
            size: u64,
            prefer_device_local: bool,
        };

        pub const InitDepthStencilFormatOptions = struct {
            name: DebugName,
            size: u64,
            format: ImageOptions.Format.DepthStencil,
        };

        pub const InitDepthStencilFormatWriteOptions = struct {
            name: DebugName,
            size: u64,
            format: ImageOptions.Format.DepthStencil,
            prefer_device_local: bool,
        };

        pub const InitOptions = if (kind.usage) |usage| switch (usage) {
            .buf => if (kind.access == .write) InitNoFormatWriteOptions else InitNoFormatOptions,
            .image => |image| if (image.format) |format| switch (format) {
                .color => if (kind.access == .write) InitNoFormatWriteOptions else InitNoFormatOptions,
                .depth_stencil => if (kind.access == .write) InitDepthStencilFormatWriteOptions else InitDepthStencilFormatOptions,
            } else @compileError("missing image format"),
        } else @compileError("missing usage");

        pub inline fn init(gx: *Ctx, options: @This().InitOptions) @This() {
            comptime assert(kind.usage != null);
            comptime switch (kind.usage.?) {
                .buf => |buffer_kind| buffer_kind.assertNonZero(),
                .image => |image| {
                    assert(image.nonNull());
                },
            };

            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();

            assert(options.size > 0);

            const untyped = Backend.memoryCreate(gx, .{
                .name = options.name,
                .usage = switch (kind.usage.?) {
                    .buf => |buf| .{ .buf = buf },
                    .image => |image| switch (image.format.?) {
                        .color => .{
                            .color_image = .{
                                .tiling = image.tiling.?,
                                .transient_attachment = image.transient_attachment.?,
                            },
                        },
                        .depth_stencil => |format| .{ .depth_stencil_image = .{
                            .tiling = image.tiling.?,
                            .transient_attachment = image.transient_attachment.?,
                            .format = format,
                        } },
                    },
                },
                .access = switch (kind.access) {
                    .read => .read,
                    .write => .{ .write = .{ .prefer_device_local = options.prefer_device_local } },
                    .none => .none,
                },
                .size = options.size,
            });
            tracy.alloc(.{
                .ptr = @ptrFromInt(@intFromEnum(untyped)),
                .size = options.size,
                .pool_name = tracy_gpu_pool,
            });
            return @enumFromInt(@intFromEnum(untyped));
        }

        pub fn deinit(self: @This(), gx: *Ctx) void {
            tracy.free(.{
                .ptr = @ptrFromInt(@intFromEnum(self)),
                .pool_name = tracy_gpu_pool,
            });
            Backend.deviceMemoryDestroy(gx, self.as(.{}));
        }

        pub inline fn as(
            self: Self,
            comptime result_kind: DeviceMemKind,
        ) Memory(result_kind) {
            kind.checkCast(result_kind);
            return @enumFromInt(@intFromEnum(self));
        }

        pub inline fn fromBackendType(value: Backend.Memory) @This() {
            comptime assert(@sizeOf(Backend.Memory) == @sizeOf(@This()));
            return @enumFromInt(@intFromEnum(value));
        }

        pub inline fn asBackendType(self: @This()) Backend.Memory {
            comptime assert(@sizeOf(Backend.Memory) == @sizeOf(@This()));
            return @enumFromInt(@intFromEnum(self));
        }

        _,
    };
}

pub const DeviceMemKind = struct {
    pub const Usage = union(enum) {
        image: ImageKind,
        buf: BufKind,

        inline fn checkCast(comptime self: ?@This(), comptime rtype: ?@This()) void {
            if (rtype == null) return;
            comptime if (self) |ltype| {
                if (std.meta.activeTag(ltype) == std.meta.activeTag(rtype.?)) {
                    switch (ltype) {
                        .image => |image| return image.checkCast(rtype.?.image),
                        .buf => |buf| return buf.checkCast(rtype.?.buf),
                    }
                }
            };
            const self_name = if (self) |s| @tagName(s) else "null";
            @compileError("cannot cast " ++ self_name ++ " to " ++ @tagName(rtype.?));
        }
    };

    pub const Access = enum {
        none,
        write,
        read,

        inline fn checkCast(comptime self: @This(), comptime rtype: @This()) void {
            if (comptime rtype != .none and self != rtype) {
                @compileError("cannot cast " ++ @tagName(self) ++ " to " ++ @tagName(rtype));
            }
        }
    };

    usage: ?Usage = null,
    access: Access = .none,

    inline fn checkCast(comptime self: @This(), comptime rtype: @This()) void {
        Usage.checkCast(self.usage, rtype.usage);
        self.access.checkCast(rtype.access);
    }
};

pub fn DeviceMemView(kind: DeviceMemKind) type {
    return struct {
        memory: Memory(kind),
        offset: u64,
        size: u64,

        pub inline fn as(
            self: @This(),
            comptime result_kind: DeviceMemKind,
        ) DeviceMemView(result_kind) {
            kind.checkCast(result_kind);
            return .{
                .memory = @enumFromInt(@intFromEnum(self.memory)),
                .offset = self.offset,
                .size = self.size,
            };
        }
    };
}

pub fn DeviceMemViewUnsized(kind: DeviceMemKind) type {
    return struct {
        memory: Memory(kind),
        offset: u64,

        pub inline fn as(
            self: @This(),
            comptime result_kind: DeviceMemKind,
        ) DeviceMemViewUnsized(result_kind) {
            kind.checkCast(result_kind);
            return .{
                .memory = self.memory.as(result_kind),
                .offset = self.offset,
            };
        }
    };
}

pub const MemoryCreateUntypedOptions = struct {
    pub const Access = union(enum) {
        none: void,
        write: struct { prefer_device_local: bool },
        read: void,

        fn asAccess(self: @This()) DeviceMemKind.Access {
            return switch (self) {
                .none => .none,
                .write => .write,
                .read => .read,
            };
        }
    };

    pub const Usage = union(enum) {
        color_image: Image(.{}).InitColorOptions.Kind,
        depth_stencil_image: Image(.{}).InitDepthStencilOptions.Kind,

        fn asUsage(self: @This()) DeviceMemKind.Usage {
            return switch (self) {
                .color_image => |image| .{ .image = .{
                    .format = .color,
                    .tiling = image.tiling,
                    .transient_attachment = image.transient_attachment,
                } },
                .depth_stencil_image => |image| .{ .image = .{
                    .format = .{ .depth_stencil = image.format },
                    .tiling = image.tiling,
                    .transient_attachment = image.transient_attachment,
                } },
            };
        }
    };

    name: DebugName,
    usage: Usage,
    access: Access = .none,
    size: u64,
};

/// Some APIs have separate handles for descriptor set layouts and pipelines (e.g. Vulkan), others
/// have a single handle that refers to both the pipeline layout state and the descriptor set layout
/// state (e.g. DirectX 12). On APIs that don't distinguish between pipeline and descriptor set
/// layouts, both handles are equivalent.
pub const CombinedPipelineLayout = struct {
    pipeline: PipelineLayout,
    descriptor_set: DescSetLayout,

    pub const InitOptions = struct {
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
        descriptors: []const Desc,

        fn descriptor(comptime self: *const @This(), comptime name: []const u8) *const Desc {
            const index = self.binding(name);
            return &self.descriptors[index];
        }

        fn binding(comptime self: *const @This(), comptime name: []const u8) u32 {
            const result = comptime for (self.descriptors, 0..) |desc, i| {
                if (std.mem.eql(u8, desc.name, name)) {
                    break i;
                }
            } else @compileError("no such binding " ++ name);
            return result;
        }

        pub fn CreateCmdOptions(self: *const @This()) type {
            return struct {
                pipeline_name: DebugName,
                shader_name: DebugName,
                stages: InitCombinedPipelineCmd.Stages,
                input_assembly: InitCombinedPipelineCmd.InputAssembly,
                result: *CombinedPipeline(self),
            };
        }

        pub inline fn createCmd(
            comptime self: *const @This(),
            gx: *const Ctx,
            options: CreateCmdOptions(self),
        ) InitCombinedPipelineCmd {
            return .{
                .pipeline_name = options.pipeline_name,
                .shader_name = options.shader_name,
                .layout = CombinedPipelineLayout.get(gx, self),
                .stages = options.stages,
                .result = @ptrCast(options.result),
                .input_assembly = options.input_assembly,
            };
        }

        pub fn DrawCmdOptions(self: *const CombinedPipelineLayout.InitOptions) type {
            return struct {
                combined_pipeline: CombinedPipeline(self),
                args: Buf(.{ .indirect = true }).UnsizedView,
                args_count: u32,
                indices: ?Buf(.{ .index = true }) = null,
                descriptor_set: DescSet(self),
            };
        }

        pub inline fn drawCmd(comptime self: *const @This(), options: @This().DrawCmdOptions(self)) DrawCmd {
            return .{
                .combined_pipeline = options.combined_pipeline.asUntyped(),
                .args = options.args,
                .args_count = options.args_count,
                .indices = options.indices,
                .descriptor_set = options.descriptor_set.asUntyped(),
            };
        }

        pub fn CreateDescSetOptions(comptime self: *const CombinedPipelineLayout.InitOptions) type {
            return struct {
                comptime layout: *const CombinedPipelineLayout.InitOptions = self,
                name: DebugName,
                result: *DescSet(self),
            };
        }

        pub fn createDescSetCmd(
            comptime self: *const @This(),
            gx: *const Ctx,
            options: @This().CreateDescSetOptions(self),
        ) DescPool.InitOptions.Cmd {
            return .{
                .name = options.name,
                .layout = CombinedPipelineLayout.get(gx, self).descriptor_set,
                .layout_create_options = self,
                .result = @ptrCast(options.result),
            };
        }

        pub fn UpdateDescOptions(self: *const @This(), comptime name: []const u8) type {
            const b = self.binding(name);
            const desc = self.descriptors[b];
            return struct {
                set: DescSet(self),
                value: switch (desc.kind) {
                    .storage_buffer => Buf(.{ .storage = true }).View,
                    .uniform_buffer => Buf(.{ .uniform = true }).UnsizedView,
                    .combined_image_sampler => DescUpdateCmd.Value.CombinedImageSampler,
                },
            };
        }

        pub inline fn updateDescCmd(
            comptime self: *const @This(),
            comptime name: []const u8,
            options: self.UpdateDescOptions(name),
        ) DescUpdateCmd {
            return self.updateDescItemCmd(name, 0, options);
        }

        pub inline fn updateDescItemCmd(
            comptime self: *const @This(),
            comptime name: []const u8,
            comptime index: u32,
            options: self.UpdateDescOptions(name),
        ) DescUpdateCmd {
            const b = comptime self.binding(name);
            const desc = self.descriptors[b];

            if (index >= self.descriptors[b].count) {
                @compileError("out of bounds update");
            }

            return .{
                .set = options.set.asUntyped(),
                .binding = b,
                .index = index,
                .value = switch (desc.kind) {
                    .uniform_buffer => |uniform_buffer| .{ .uniform_buffer_view = .{
                        .buf = options.value.buf.as(.{}),
                        .offset = options.value.offset,
                        .size = uniform_buffer.size,
                    } },
                    .storage_buffer => .{ .storage_buffer_view = options.value.as(.{}) },
                    .combined_image_sampler => .{ .combined_image_sampler = options.value },
                },
            };
        }
    };

    pub fn init(
        gx: *Ctx,
        comptime max_descriptors: u32,
        options: @This().InitOptions,
    ) CombinedPipelineLayout {
        return Backend.combinedPipelineLayoutCreate(gx, max_descriptors, options);
    }

    pub fn deinit(self: @This(), gx: *Ctx) void {
        Backend.combinedPipelineLayoutDestroy(gx, self);
    }

    pub fn get(self: *const Ctx, comptime kind: *const @This().InitOptions) @This() {
        inline for (global_options.combined_pipeline_layouts, 0..) |curr, i| {
            if (curr == kind) return self.combined_pipeline_layout_typed[i];
        } else @compileError("layout kind not registered in global options");
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

pub const Pipeline = enum(u64) {
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

pub const InitPipelinesError = error{
    Layout,
};

pub const InitCombinedPipelineCmd = struct {
    pub const Stages = struct {
        pub const max_stages = std.meta.fields(@This()).len;
        const ShaderModule = struct {
            spv: []const u32,
            name: DebugName,
        };
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

    pipeline_name: DebugName,
    shader_name: DebugName,
    layout: CombinedPipelineLayout,
    stages: Stages,
    result: *CombinedPipeline(null),
    input_assembly: InputAssembly,
};

pub fn CombinedPipeline(init_options: ?*const CombinedPipelineLayout.InitOptions) type {
    return struct {
        const type_dependency = init_options;

        pipeline: Pipeline,
        layout: PipelineLayout,

        pub fn deinit(self: @This(), gx: *Ctx) void {
            Backend.combinedPipelineDestroy(gx, self.asUntyped());
        }

        pub inline fn asUntyped(self: @This()) CombinedPipeline(null) {
            return .{
                .pipeline = self.pipeline,
                .layout = self.layout,
            };
        }
    };
}

pub fn initCombinedPipelines(
    self: *@This(),
    comptime max_cmds: u32,
    cmds: []const InitCombinedPipelineCmd,
) InitPipelinesError!void {
    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();
    if (cmds.len == 0) return;
    Backend.combinedPipelinesCreate(self, max_cmds, cmds);
}

pub const Sampler = enum(u64) {
    _,

    pub const InitOptions = struct {
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

        name: DebugName,
        mag_filter: Filter,
        min_filter: Filter,
        mipmap_mode: Filter,
        address_mode: AddressModes,
        mip_lod_bias: f32,
        max_anisotropy_hint: f32,
        compare_op: ?CompareOp,
        min_lod: f32,
        max_lod: ?f32,
        border_color: BorderColor,
        unnormalized_coordinates: bool,
    };

    pub fn init(gx: *Ctx, options: @This().InitOptions) Sampler {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();
        if (std.debug.runtime_safety) {
            const ma = options.max_anisotropy_hint;
            assert(ma == 0.0 or (!std.math.isNan(ma) and ma >= 1.0));
        }
        return Backend.samplerCreate(gx, options);
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

pub const TransferCmd = union(enum) {
    const CopyBufferToBuffer = struct {
        const Region = struct {
            src_offset: u64,
            dst_offset: u64,
            size: u64,
        };

        src: Buf(.{ .transfer_src = true }),
        dst: Buf(.{ .transfer_dst = true }),
        regions: []const Region,
    };

    const CopyBufferToColorImage = struct {
        const Region = struct {
            buffer_offset: u64 = 0,
            buffer_row_length: ?u32 = null,
            buffer_image_height: ?u32 = null,
            mip_level: u32 = 0,
            base_array_layer: u32 = 0,
            layer_count: u32 = 1,
            image_offset: Offset = .{ .x = 0, .y = 0, .z = 0 },
            image_extent: ImageOptions.Extent,
        };

        const Offset = struct { x: i32, y: i32, z: i32 };

        buf: Buf(.{ .transfer_src = true }),
        image: Image(.{ .format = .color }),
        base_mip_level: u32 = 0,
        level_count: u32 = 1,
        base_array_layer: u32 = 0,
        layer_count: u32 = 1,
        new_layout: ImageOptions.Layout,
        regions: []const Region,
    };

    copy_buffer_to_buffer: CopyBufferToBuffer,
    copy_buffer_to_color_image: CopyBufferToColorImage,
};

pub fn waitIdle(self: *const @This()) void {
    Backend.waitIdle(self);
}

fn containsBits(self: anytype, other: @TypeOf(self)) bool {
    const Int = std.meta.Int(.unsigned, @bitSizeOf(@TypeOf(self)));
    const self_bits: Int = @bitCast(self);
    const other_bits: Int = @bitCast(other);
    return self_bits & other_bits == other_bits;
}
