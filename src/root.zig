//! A light graphics API abstraction.
//!
//! See the README for more information.

const std = @import("std");
const assert = std.debug.assert;
const root = @import("root");
const BufView = @import("buf_view.zig").BufView;
const Backend = global_options.Backend;
const TracyQueue = tracy.GpuQueue;
const Zone = tracy.Zone;

pub const tracy = @import("tracy");

pub const ext = @import("ext.zig");

pub const tracy_gpu_pool = "gpu";

/// Compile time options for the library. You must declare a constant of this type in your root file
/// named `gpu_options` to configure the library.
pub const Options = struct {
    Backend: type,
    max_frames_in_flight: u4 = 2,
    blocking_zone_color: tracy.Color = .dark_sea_green4,
    init_pipelines_buf_len: u32 = 16,
    init_desc_pool_buf_len: u32 = 16,
    update_desc_sets_buf_len: u32 = 32,
    combined_pipeline_layout_create_buf_len: u32 = 16,
};

pub const global_options: Options = root.gpu_options;

pub const Gx = @import("Gx.zig");
pub const Writer = @import("Writer.zig");

pub const btypes = @import("btypes.zig");

const gpu = @This();

pub const Extent2D = struct {
    pub const zero: @This() = .{ .width = 0, .height = 0 };
    width: u32,
    height: u32,
};

pub const Offset2D = struct {
    pub const zero: @This() = .{ .x = 0, .y = 0 };
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
    dedicated: DedicatedAllocationAffinity,

    /// Bumps the given offset by these memory requirements. If a dedicated allocation is preferred
    /// or required, the offset is left unchanged.
    pub fn bump(self: @This(), offset: *u64) void {
        if (self.dedicated != .discouraged) return;
        offset.* = std.mem.alignForward(u64, offset.*, self.alignment);
        offset.* += self.size;
    }
};

pub const DebugName = struct {
    str: [*:0]const u8,
    index: ?usize = null,

    pub fn format(
        self: @This(),
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("{s}", .{self.str});
        if (self.index) |index| {
            try writer.print(" {}", .{index});
        }
    }
};

pub fn Buf(k: BufKind) type {
    return struct {
        pub const kind = k;

        memory: MemoryHandle,
        handle: BufHandle(kind),
        size: u64,

        pub const View = BufView(@This());

        pub const Options = struct {
            name: DebugName,
            size: u64,
        };

        pub inline fn init(
            gx: *Gx,
            options: @This().Options,
        ) Buf(kind) {
            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();
            comptime kind.assertNonZero();
            assert(options.size > 0); // Vulkan doesn't support zero sized buffers
            const untyped = Backend.bufCreate(gx, options.name, kind, options.size);
            tracy.alloc(.{
                .ptr = @ptrFromInt(@intFromEnum(untyped.memory)),
                .size = untyped.size,
                .pool_name = tracy_gpu_pool,
            });
            return .{
                .memory = @enumFromInt(@intFromEnum(untyped.memory)),
                .handle = @enumFromInt(@intFromEnum(untyped.handle)),
                .size = untyped.size,
            };
        }

        pub fn deinit(self: @This(), gx: *Gx) void {
            self.handle.deinit(gx);
            self.memory.deinit(gx);
        }

        pub inline fn as(self: @This(), comptime result_kind: BufKind) Buf(result_kind) {
            return .{
                .memory = self.memory,
                .handle = self.handle.as(result_kind),
                .size = self.size,
            };
        }

        pub fn view(self: @This()) View {
            return .{
                .handle = self.handle,
                .ptr = {},
                .len = self.size,
                .offset = 0,
            };
        }
    };
}

pub fn ReadbackBuf(k: BufKind) type {
    return struct {
        pub const kind = k;

        memory: MemoryHandle,
        handle: BufHandle(kind),
        data: []const u8,

        pub const View = BufView(@This());

        pub const Options = struct {
            name: DebugName,
            size: u64,
        };

        pub inline fn init(
            gx: *Gx,
            options: @This().Options,
        ) @This() {
            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();
            comptime kind.assertNonZero();
            const untyped = Backend.readbackBufCreate(gx, options.name, kind, options.size);
            tracy.alloc(.{
                .ptr = @ptrFromInt(@intFromEnum(untyped.memory)),
                .size = untyped.data.len,
                .pool_name = tracy_gpu_pool,
            });
            return .{
                .memory = @enumFromInt(@intFromEnum(untyped.memory)),
                .handle = @enumFromInt(@intFromEnum(untyped.handle)),
                .data = untyped.data,
            };
        }

        pub fn deinit(self: @This(), gx: *Gx) void {
            self.handle.deinit(gx);
            self.memory.deinit(gx);
        }

        pub inline fn as(self: @This(), comptime result_kind: BufKind) ReadbackBuf(result_kind) {
            return .{
                .memory = self.memory.as(.{ .access = .read, .usage = .{ .buf = result_kind } }),
                .buf = self.buf.as(result_kind),
                .ptr = self.ptr,
                .size = self.size,
            };
        }

        pub inline fn asBuf(self: @This(), comptime result_kind: BufKind) Buf(result_kind) {
            return .{
                .memory = self.memory,
                .handle = self.handle.as(result_kind),
                .size = self.data.len,
            };
        }

        pub fn view(self: @This()) View {
            return .{
                .handle = self.handle,
                .ptr = self.data.ptr,
                .len = self.data.len,
                .offset = 0,
            };
        }
    };
}

pub fn UploadBuf(k: BufKind) type {
    return struct {
        pub const kind = k;

        memory: MemoryHandle,
        handle: BufHandle(kind),
        data: []volatile anyopaque,

        pub const View = BufView(@This());

        pub const Options = struct {
            name: DebugName,
            size: u64,
            prefer_device_local: bool,
        };

        pub inline fn init(
            gx: *Gx,
            options: @This().Options,
        ) UploadBuf(kind) {
            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();
            comptime kind.assertNonZero();
            const untyped = Backend.uploadBufCreate(
                gx,
                options.name,
                kind,
                options.size,
                options.prefer_device_local,
            );
            tracy.alloc(.{
                .ptr = @ptrFromInt(@intFromEnum(untyped.memory)),
                .size = untyped.data.len,
                .pool_name = tracy_gpu_pool,
            });
            return .{
                .memory = @enumFromInt(@intFromEnum(untyped.memory)),
                .handle = @enumFromInt(@intFromEnum(untyped.handle)),
                .data = untyped.data,
            };
        }

        pub fn deinit(self: @This(), gx: *Gx) void {
            self.handle.deinit(gx);
            self.memory.deinit(gx);
        }

        pub inline fn as(self: @This(), comptime result_kind: BufKind) UploadBuf(result_kind) {
            return .{
                .memory = self.memory,
                .handle = self.handle.as(result_kind),
                .data = self.data,
            };
        }

        pub inline fn asBuf(self: @This(), comptime result_kind: BufKind) Buf(result_kind) {
            return .{
                .memory = self.memory,
                .handle = self.handle.as(result_kind),
                .size = self.data.len,
            };
        }

        pub fn view(self: @This()) View {
            return .{
                .handle = self.handle,
                .ptr = self.data.ptr,
                .len = self.data.len,
                .offset = 0,
            };
        }
    };
}

pub const BufKind = packed struct {
    transfer_src: bool = false,
    transfer_dst: bool = false,
    uniform_texel: bool = false,
    storage_texel: bool = false,
    uniform: bool = false,
    storage: bool = false,
    index: bool = false,
    indirect: bool = false,

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

pub fn BufHandle(kind: BufKind) type {
    return enum(u64) {
        const Self = @This();

        pub const Options = struct {
            name: DebugName,
            size: u64,
        };

        pub inline fn as(self: Self, comptime result_kind: BufKind) BufHandle(result_kind) {
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

        pub fn deinit(self: @This(), gx: *Gx) void {
            Backend.bufDestroy(gx, self.as(.{}));
        }

        _,
    };
}

pub const MemoryHandle = enum(u64) {
    _,

    pub fn deinit(self: @This(), gx: *Gx) void {
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
        handle: MemoryHandle,
        size: u64,

        pub const kind = k;

        pub const Options = struct {
            name: DebugName,
            size: u64,
        };

        pub inline fn init(gx: *Gx, options: @This().Options) @This() {
            comptime switch (kind) {
                .buf => |buffer_kind| buffer_kind.assertNonZero(),
                .color_image, .depth_stencil_image => {},
                .any => unreachable,
            };

            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();

            assert(options.size > 0);

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
                .handle = @enumFromInt(@intFromEnum(any)),
                .size = options.size,
            };
        }

        pub fn deinit(self: @This(), gx: *Gx) void {
            self.handle.deinit(gx);
        }

        pub inline fn as(
            self: @This(),
            comptime result_kind: MemoryKind,
        ) Memory(result_kind) {
            MemoryKind.checkCast(kind, result_kind);
            return .{
                .handle = self.handle,
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

fn containsBits(self: anytype, other: @TypeOf(self)) bool {
    const Int = std.meta.Int(.unsigned, @bitSizeOf(@TypeOf(self)));
    const self_bits: Int = @bitCast(self);
    const other_bits: Int = @bitCast(other);
    return self_bits & other_bits == other_bits;
}

// Good reference for support on DX12 level hardware:
// - https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/hardware-support-for-direct3d-12-1-formats
pub const ImageFormat = enum(i32) {
    undefined = Backend.named_image_formats.undefined,

    r8g8b8a8_unorm = Backend.named_image_formats.r8g8b8a8_unorm,
    r8g8b8a8_snorm = Backend.named_image_formats.r8g8b8a8_snorm,
    r8g8b8a8_uscaled = Backend.named_image_formats.r8g8b8a8_uscaled,
    r8g8b8a8_sscaled = Backend.named_image_formats.r8g8b8a8_sscaled,
    r8g8b8a8_uint = Backend.named_image_formats.r8g8b8a8_uint,
    r8g8b8a8_sint = Backend.named_image_formats.r8g8b8a8_sint,

    r8g8b8a8_srgb = Backend.named_image_formats.r8g8b8a8_srgb,

    d24_unorm_s8_uint = Backend.named_image_formats.d24_unorm_s8_uint,
    d32_sfloat = Backend.named_image_formats.d32_sfloat,

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

    pub fn deinit(self: @This(), gx: *Gx) void {
        Backend.imageDestroy(gx, self);
    }
};

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
                        .input_attachment = false,
                    },
                    .any => unreachable,
                },
                .aspect = switch (kind) {
                    .depth_stencil => |ds| ds.aspect,
                    .color => .{ .color = true },
                    .any => unreachable,
                },
                .mip_levels = options.mip_levels,
                .array_layers = options.array_layers,
            };
        }

        pub const ColorOptions = struct {
            const Usage = packed struct {
                transfer_src: bool = false,
                transfer_dst: bool = false,
                sampled: bool = false,
                storage: bool = false,
                color_attachment: bool = false,
            };

            flags: ImageFlags = .{},
            dimensions: Dimensions = .@"2d",
            format: ImageFormat,
            extent: ImageExtent,
            samples: Samples = .@"1",
            usage: Usage,
            mip_levels: u32 = 1,
            array_layers: u32 = 1,

            pub fn memoryRequirements(self: @This(), gx: *Gx) MemoryRequirements {
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
            };

            pub fn memoryRequirements(self: @This(), gx: *Gx) MemoryRequirements {
                return Backend.imageMemoryRequirements(gx, backendOptions(self));
            }

            flags: ImageFlags = .{},
            dimensions: Dimensions = .@"2d",
            extent: ImageExtent,
            samples: Samples = .@"1",
            usage: Usage,
            aspect: ImageAspect,
            mip_levels: u32 = 1,
            array_layers: u32 = 1,
        };

        pub const Options = switch (kind) {
            .color => ColorOptions,
            .depth_stencil => DepthStencilOptions,
            .any => unreachable,
        };

        pub const InitDedicatedOptions = struct {
            name: DebugName,
            image: Image(kind).Options,
        };

        pub const InitDedicatedResult = struct {
            memory: Memory(kind.asMemoryKind()),
            image: Image(kind),
        };

        /// Creates the image with a dedicated allocation.
        pub fn initDedicated(gx: *Gx, options: InitDedicatedOptions) InitDedicatedResult {
            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();
            const untyped = Backend.imageCreateDedicated(
                gx,
                options.name,
                Image(kind).backendOptions(options.image),
            );
            tracy.alloc(.{
                .ptr = @ptrFromInt(@intFromEnum(untyped.memory.handle)),
                .size = untyped.memory.size,
                .pool_name = tracy_gpu_pool,
            });
            return .{
                .memory = .{
                    .handle = untyped.memory.handle,
                    .size = untyped.memory.size,
                },
                .image = .{
                    .handle = untyped.image.handle,
                    .view = untyped.image.view,
                },
            };
        }

        pub const InitPlacedOptions = struct {
            name: DebugName,
            memory: Memory(kind.asMemoryKind()),
            offset: u64,
            image: Image(kind).Options,
        };

        /// Place the image beginning at the start of the given memory view. The caller is
        /// responsible for ensuring proper alignment, not overrunning the buffer, and checking
        /// that this image does not require a dedicated allocation on the current hardware.
        pub fn initPlaced(gx: *Gx, options: InitPlacedOptions) @This() {
            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();
            if (options.offset > options.memory.size) @panic("OOB");
            const result = Backend.imageCreatePlaced(
                gx,
                options.name,
                options.memory.handle,
                options.offset,
                Image(kind).backendOptions(options.image),
            );
            return .{
                .handle = result.handle,
                .view = result.view,
            };
        }

        pub fn deinit(self: @This(), gx: *Gx) void {
            self.view.deinit(gx);
            self.handle.deinit(gx);
        }

        pub fn asAny(self: @This()) Image(.any) {
            return .{
                .handle = @enumFromInt(@intFromEnum(self.handle)),
                .view = self.view,
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

    pub fn deinit(self: @This(), gx: *Gx) void {
        Backend.imageViewDestroy(gx, self);
    }
};

pub const ImageKind = union(enum) {
    color,
    depth_stencil: ImageFormat,
    any,

    pub fn asMemoryKind(comptime self: @This()) MemoryKind {
        return switch (self) {
            .color => .color_image,
            .depth_stencil => |format| .{ .depth_stencil_image = format },
            .any => .any,
        };
    }

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
};

pub const ShaderModule = enum(u64) {
    _,

    pub const Options = struct {
        name: DebugName,
        ir: []const u32,
    };

    pub fn init(gx: *Gx, options: @This().Options) @This() {
        return Backend.shaderModuleCreate(gx, options);
    }

    pub fn deinit(self: @This(), gx: *Gx) void {
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
    handle: Handle,
    layout: Layout.Handle,
    kind: Kind,

    pub const Kind = enum {
        graphics,
        compute,
    };

    pub const Handle = enum(u64) {
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

    pub const InitGraphicsCmd = struct {
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
        layout: Layout,
        stages: Stages,
        result: *Pipeline,
        input_assembly: InputAssembly,
        color_attachment_formats: []const ImageFormat,
        depth_attachment_format: ImageFormat,
        stencil_attachment_format: ImageFormat,
    };

    pub fn initGraphics(
        gx: *Gx,
        cmds: []const InitGraphicsCmd,
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
        Backend.pipelinesCreateGraphics(gx, cmds);
    }

    pub const InitComputeCmd = struct {
        name: DebugName,
        layout: Layout,
        result: *Pipeline,
        shader_module: ShaderModule,
    };

    pub fn initCompute(
        gx: *Gx,
        cmds: []const InitComputeCmd,
    ) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();
        Backend.pipelinesCreateCompute(gx, cmds);
    }

    pub fn deinit(self: @This(), gx: *Gx) void {
        Backend.pipelineDestroy(gx, self);
    }

    /// Some APIs have separate handles for descriptor set layouts and pipelines (e.g. Vulkan), others
    /// have a single handle that refers to both the pipeline layout state and the descriptor set layout
    /// state (e.g. DireGx 12). On APIs that don't distinguish between pipeline and descriptor set
    /// layouts, both handles are equivalent.
    pub const Layout = struct {
        handle: @This().Handle,
        desc_set: DescSet.Layout,

        pub const Handle = enum(u64) {
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
                    storage_image: void,
                };
                pub const Stages = packed struct {
                    vertex: bool = false,
                    fragment: bool = false,
                    compute: bool = false,
                };

                name: []const u8,
                kind: Desc.Kind,
                count: u32 = 1,
                stages: Stages,
                /// Careful. If false, all descriptors must be bound, but at the time of writing the
                /// Vulkan validation layers will not catch if you fail to set this flag.
                ///
                /// Also note that if indexing the descriptor very often (e.g. once per pixel), this
                /// can affect GPU assisted validation performance. Some drivers have buggy caching
                /// which can make this performance degradation "sticky", i.e. you may need to edit
                /// your shaders to invalidate the cache to get expected perf back while GPU
                /// assisted validation is still enabled.
                partially_bound: bool,
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

        pub fn init(gx: *Gx, options: @This().Options) Layout {
            return Backend.pipelineLayoutCreate(gx, options);
        }

        pub fn deinit(self: @This(), gx: *Gx) void {
            Backend.pipelineLayoutDestroy(gx, self);
        }
    };
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

    pub fn init(gx: *Gx, name: DebugName, options: @This().Options) Sampler {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();
        return Backend.samplerCreate(gx, name, options);
    }

    pub fn deinit(self: @This(), gx: *Gx) void {
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

pub const DescPool = enum(u64) {
    _,

    pub const Options = struct {
        pub const Cmd = struct {
            name: DebugName,
            layout: DescSet.Layout,
            layout_options: *const Pipeline.Layout.Options,
            result: *DescSet,
        };
        name: DebugName,
        cmds: []const Cmd,
    };

    pub fn init(gx: *Gx, options: @This().Options) @This() {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();
        return Backend.descPoolCreate(gx, options);
    }

    pub fn deinit(self: @This(), gx: *Gx) void {
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

pub const DescSet = enum(u64) {
    _,

    pub const Update = struct {
        pub const Value = union(enum) {
            pub const Tag: type = @typeInfo(@This()).@"union".tag_type.?;
            pub const CombinedImageSampler = struct {
                pub const Layout = enum {
                    read_only,
                    attachment,
                };

                view: ImageView,
                sampler: Sampler,
                layout: @This().Layout,
            };

            storage_buf: Buf(.{ .storage = true }).View,
            uniform_buf: Buf(.{ .uniform = true }).View,
            combined_image_sampler: CombinedImageSampler,
            storage_image: ImageView,
        };

        set: DescSet,
        binding: u32,
        // The size of this integer is conservative, the backend APIs typically accept `u32`s here.
        // However, while they accept `u32`s, hardware places additional limits on how many
        // resources of various types can be passed in. Notably, Vulkan's Roadmap to 2022 only
        // guarantees `maxDescriptorSetSampledImages` to be 1800. Setting this index to a smaller
        // type is just a quick smoke test against going too far past these limits, it's still
        // possible to violate them by creating multiple descriptor sets.
        //
        // If you need 32 bit indices here, you can replace the `u10` with a `u32`. Feel free to
        // open a PR if you do and I'll reconsider this limit.
        index: u10 = 0,
        value: Value,
    };

    pub const Layout = enum(u64) {
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

    pub inline fn fromBackendType(value: Backend.DescSet) @This() {
        comptime assert(@sizeOf(Backend.DescSet) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.DescSet {
        comptime assert(@sizeOf(Backend.DescSet) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }

    /// Submit descriptor set update commands. Fastest when sorted. Copy commands not currently
    /// supported.
    pub fn update(gx: *Gx, cmds: []const Update) void {
        Backend.descSetsUpdate(gx, cmds);
    }
};

pub const Device = struct {
    // https://registry.khronos.org/vulkan/specs/1.3/html/chap33.html#limits-minmax
    pub const StorageBufSize = u27;

    // https://registry.khronos.org/vulkan/specs/1.3/html/chap33.html#limits-minmax
    pub const max_uniform_buf_offset_alignment = 256;
    // https://registry.khronos.org/vulkan/specs/1.3/html/chap33.html#limits-minmax
    pub const max_storage_buf_offset_alignment = 256;

    /// The required alignment of a resource in a buffer being copied.
    ///
    /// When running Vulkan, this value is just a recommendation, and can vary per device--real GPUs
    /// sometimes set it to 1. With DX12, it's a requirement, and is always 512. We're exposing the DX12
    /// value since it's the strictest.
    ///
    /// https://learn.microsoft.com/en-us/windows/win32/direct3d12/upload-and-readback-of-texture-data
    pub const buffer_copy_offset_alignment = 512;

    /// The required row pitch alignment of a resource in a buffer being copied. Keep in mind that the
    /// row pitch is the distance between the rows, this means that tightly packed rows are always
    /// properly aligned, allowing you to ignore this value if you like.
    ///
    /// See `buffer_copy_offset_alignment` for more info on why this is a constant. This is also often
    /// 1 on real GPUs under Vulkan.
    ///
    /// https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_subresource_footprint
    pub const buffer_copy_row_pitch_alignment = 256;

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
    texel_buffer_offset_alignment: u16,
    timestamp_period: f32,
    tracy_queue: TracyQueue,
    surface_format: ImageFormat,
};

pub const ImageBarrier = extern struct {
    backend: Backend.ImageBarrier,

    pub const Range = struct {
        pub const first: @This() = .{
            .base_mip_level = 0,
            .mip_levels = 1,
            .base_array_layer = 0,
            .array_layers = 1,
        };
        base_mip_level: u32,
        mip_levels: u32,
        base_array_layer: u32,
        array_layers: u32,
    };

    pub const UndefinedToTransferDstOptions = struct {
        handle: ImageHandle,
        range: Range,
        aspect: ImageAspect,
    };

    pub fn undefinedToTransferDst(options: UndefinedToTransferDstOptions) @This() {
        return Backend.imageBarrierUndefinedToTransferDst(options);
    }

    pub const UndefinedToColorAttachmentOptions = struct {
        handle: ImageHandle,
        range: Range,
    };

    pub fn undefinedToColorAttachment(options: UndefinedToColorAttachmentOptions) @This() {
        return Backend.imageBarrierUndefinedToColorAttachment(options);
    }

    pub const UndefinedToColorAttachmentOptionsAfterRead = struct {
        pub const Stage = packed struct {
            vertex_shader: bool = false,
            fragment_shader: bool = false,
        };

        handle: ImageHandle,
        range: Range,
        src_stage: Stage,
    };

    pub fn undefinedToColorAttachmentAfterRead(options: UndefinedToColorAttachmentOptionsAfterRead) @This() {
        return Backend.imageBarrierUndefinedToColorAttachmentAfterRead(options);
    }

    pub const TransferDstToReadOnlyOptions = struct {
        pub const Stage = packed struct {
            vertex_shader: bool = false,
            fragment_shader: bool = false,
        };

        handle: ImageHandle,
        range: Range,
        dst_stage: Stage,
        aspect: ImageAspect,
    };

    pub fn transferDstToReadOnly(options: TransferDstToReadOnlyOptions) @This() {
        return Backend.imageBarrierTransferDstToReadOnly(options);
    }

    pub const TransferDstToColorAttachmentOptions = struct {
        handle: ImageHandle,
        range: Range,
    };

    pub fn transferDstToColorAttachment(options: TransferDstToColorAttachmentOptions) @This() {
        return Backend.imageBarrierTransferDstToColorAttachment(options);
    }

    pub const ReadOnlyToColorAttachmentOptions = struct {
        pub const Stage = packed struct {
            vertex_shader: bool = false,
            fragment_shader: bool = false,
        };

        handle: ImageHandle,
        range: Range,
        src_stage: Stage,
    };

    pub fn readOnlyToColorAttachment(options: ReadOnlyToColorAttachmentOptions) @This() {
        return Backend.imageBarrierReadOnlyToColorAttachment(options);
    }

    pub const ColorAttachmentToReadOnlyOptions = struct {
        pub const Stage = packed struct {
            vertex_shader: bool = false,
            fragment_shader: bool = false,
        };

        handle: ImageHandle,
        range: Range,
        dst_stage: Stage,
    };

    pub fn colorAttachmentToReadOnly(options: ColorAttachmentToReadOnlyOptions) @This() {
        return Backend.imageBarrierColorAttachmentToReadOnly(options);
    }

    pub const ColorAttachmentToComputeOptions = struct {
        pub const Access = struct {
            read: bool,
            write: bool,
        };
        handle: ImageHandle,
        range: Range,
        dst_access: Access,
    };

    pub fn colorAttachmentToCompute(options: ColorAttachmentToComputeOptions) @This() {
        return Backend.imageBarrierColorAttachmentToCompute(options);
    }

    pub const ComputeToColorAttachmentOptions = struct {
        pub const Access = struct {
            read: bool,
            write: bool,
        };
        handle: ImageHandle,
        range: Range,
        src_access: Access,
    };

    pub fn computeToColorAttachment(options: ComputeToColorAttachmentOptions) @This() {
        return Backend.imageBarrierComputeToColorAttachment(options);
    }

    pub const ComputeToReadOnlyOptions = struct {
        pub const Stage = packed struct {
            vertex_shader: bool = false,
            fragment_shader: bool = false,
        };
        pub const Access = struct {
            read: bool,
            write: bool,
        };
        handle: ImageHandle,
        range: Range,
        src_access: Access,
        dst_stage: Stage,
        aspect: ImageAspect,
    };

    pub fn computeToReadOnly(options: ComputeToReadOnlyOptions) @This() {
        return Backend.imageBarrierComputeToReadOnly(options);
    }

    pub const asBackendSlice = AsBackendSlice(@This()).mixin;
};

pub const BufBarrier = extern struct {
    backend: Backend.BufBarrier,

    pub const Access = union(enum) {
        compute_write: void,
        compute_read: void,
    };

    pub const ComputeWriteToGraphicsReadOptions = struct {
        const Stage = packed struct {
            vertex_shader: bool = false,
            fragment_shader: bool = false,
        };
        dst_stage: Stage,
        handle: BufHandle(.{}),
    };

    pub fn computeWriteToGraphicsRead(options: ComputeWriteToGraphicsReadOptions) @This() {
        return Backend.bufBarrierComputeWriteToGraphicsRead(options);
    }

    pub const ComputeReadToGraphicsWriteOptions = struct {
        const Stage = packed struct {
            vertex_shader: bool = false,
            fragment_shader: bool = false,
        };
        dst_stage: Stage,
        handle: BufHandle(.{}),
    };

    pub fn computeReadToGraphicsWrite(options: ComputeReadToGraphicsWriteOptions) @This() {
        return Backend.bufBarrierComputeReadToGraphicsWrite(options);
    }

    pub const GraphicsWriteToComputeReadOptions = struct {
        const Stage = packed struct {
            vertex_shader: bool = false,
            fragment_shader: bool = false,
        };
        src_stage: Stage,
        handle: BufHandle(.{}),
    };

    pub fn graphicsWriteToComputeRead(options: GraphicsWriteToComputeReadOptions) @This() {
        return Backend.bufBarrierGraphicsWriteToComputeRead(options);
    }

    pub const GraphicsReadToComputeWriteOptions = struct {
        const Stage = packed struct {
            vertex_shader: bool = false,
            fragment_shader: bool = false,
        };
        src_stage: Stage,
        handle: BufHandle(.{}),
    };

    pub fn graphicsReadToComputeWrite(options: GraphicsReadToComputeWriteOptions) @This() {
        return Backend.bufBarrierGraphicsReadToComputeWrite(options);
    }

    pub const asBackendSlice = AsBackendSlice(@This()).mixin;
};

pub const ImageUpload = struct {
    pub const Region = extern struct {
        pub const Options = struct {
            aspect: ImageAspect,
            buffer_offset: u64,
            buffer_row_length: ?u32 = null,
            buffer_image_height: ?u32 = null,
            mip_level: u32 = 0,
            base_array_layer: u32 = 0,
            array_layers: u32 = 1,
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
    src: BufHandle(.{ .transfer_src = true }),
    base_mip_level: u32,
    mip_levels: u32,
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
        pub fn next(gx: *Gx) @This() {
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
        gx: *Gx,
        comptime loc: tracy.SourceLocation.InitOptions,
    ) @This() {
        return .initFromPtr(gx, .init(loc));
    }

    pub fn initFromPtr(
        gx: *Gx,
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

    pub fn beginRendering(self: @This(), gx: *Gx, options: BeginRenderingOptions) void {
        Backend.cmdBufBeginRendering(gx, self, options);
        Backend.cmdBufSetViewport(gx, self, options.viewport);
        Backend.cmdBufSetScissor(gx, self, options.scissor);
    }

    pub fn endRendering(self: @This(), gx: *Gx) void {
        Backend.cmdBufEndRendering(gx, self);
    }

    pub fn setViewport(self: @This(), gx: *Gx, viewport: Viewport) void {
        Backend.cmdBufSetViewport(gx, self, viewport);
    }

    pub fn setScissor(self: @This(), gx: *Gx, scissor: Extent2D) void {
        Backend.cmdBufSetScissor(gx, self, scissor);
    }

    pub fn submit(self: @This(), gx: *Gx) void {
        const zone = Zone.begin(.{ .src = @src() });
        defer zone.end();
        assert(gx.in_frame);
        Backend.cmdBufSubmit(gx, self);
    }

    pub fn bindPipeline(
        self: @This(),
        gx: *Gx,
        pipeline: Pipeline,
    ) void {
        const zone = Zone.begin(.{ .src = @src() });
        defer zone.end();
        Backend.cmdBufBindPipeline(gx, self, pipeline);
    }

    pub fn bindDescSet(
        self: @This(),
        gx: *Gx,
        pipeline: Pipeline,
        set: DescSet,
    ) void {
        const zone = Zone.begin(.{ .src = @src() });
        defer zone.end();
        Backend.cmdBufBindDescSet(gx, self, pipeline, set);
    }

    pub const DrawOptions = struct {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    };

    pub fn draw(self: @This(), gx: *Gx, options: DrawOptions) void {
        const zone = Zone.begin(.{ .src = @src() });
        defer zone.end();
        Backend.cmdBufDraw(gx, self, options);
    }

    pub const DispatchOptions = struct {
        x: u32,
        y: u32,
        z: u32,
    };

    pub fn dispatch(self: @This(), gx: *Gx, options: DispatchOptions) void {
        const zone = Zone.begin(.{ .src = @src() });
        defer zone.end();
        Backend.cmdBufDispatch(gx, self, options);
    }

    pub const BarriersOptions = struct {
        image: []const ImageBarrier = &.{},
        buffer: []const BufBarrier = &.{},
    };

    pub fn barriers(self: @This(), gx: *Gx, options: BarriersOptions) void {
        Backend.cmdBufBarriers(gx, self, options);
    }

    pub fn uploadImage(self: @This(), gx: *Gx, options: ImageUpload) void {
        Backend.cmdBufUploadImage(
            gx,
            self,
            options.dst,
            options.src.as(.{}),
            options.regions,
        );
    }

    pub const UploadBufferOptions = struct {
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

        dst: BufHandle(.{ .transfer_dst = true }),
        src: BufHandle(.{ .transfer_src = true }),
        regions: []const Region,
    };

    pub fn uploadBuffer(self: @This(), gx: *Gx, options: UploadBufferOptions) void {
        Backend.cmdBufUploadBuffer(
            gx,
            self,
            options.dst.as(.{}),
            options.src.as(.{}),
            options.regions,
        );
    }

    pub fn beginZone(self: @This(), gx: *Gx, comptime loc: tracy.SourceLocation.InitOptions) void {
        self.beginZoneFromPtr(gx, .init(loc));
    }

    pub fn beginZoneFromPtr(self: @This(), gx: *Gx, loc: *const tracy.SourceLocation) void {
        Backend.cmdBufBeginZone(gx, self, loc);
    }

    pub fn endZone(self: @This(), gx: *Gx) void {
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

test {
    _ = ext.colors;
}
