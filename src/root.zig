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
const log = std.log.scoped(.gpu);

pub const tracy = @import("tracy");

pub const ext = @import("ext.zig");

pub const tracy_gpu_pool = "gpu";

/// Compile time options for the library. You must declare a constant of this type in your root file
/// named `gpu_options` to configure the library.
pub const Options = struct {
    Backend: type,
    max_frames_in_flight: u5 = 2,
    blocking_zone_color: tracy.Color = .dim_gray,
};

pub const global_options: Options = root.gpu_options;

pub const Gx = @import("Gx.zig");
pub const Writer = @import("Writer.zig");

pub const btypes = @import("btypes.zig");

const gpu = @This();

pub const Extent2D = extern struct {
    pub const zero: @This() = .{ .width = 0, .height = 0 };
    width: u32,
    height: u32,

    pub fn eql(self: @This(), other: @This()) bool {
        return std.meta.eql(self, other);
    }
};

pub const Extent3D = extern struct {
    pub const zero: @This() = .{ .width = 0, .height = 0 };
    width: u32,
    height: u32,
    depth: u32,

    pub fn eql(self: @This(), other: @This()) bool {
        return std.meta.eql(self, other);
    }
};

pub const Volume = struct {
    min: Offset3D,
    max: Offset3D,
    pub fn fromExtent2D(extent: Extent2D) @This() {
        return .{
            .min = .zero,
            .max = .{
                .x = @intCast(extent.width),
                .y = @intCast(extent.height),
                .z = 1,
            },
        };
    }
};

pub const Offset2D = extern struct {
    pub const zero: @This() = .{ .x = 0, .y = 0 };
    x: i32,
    y: i32,

    pub fn eql(self: @This(), other: @This()) bool {
        return std.meta.eql(self, other);
    }
};

pub const Offset3D = extern struct {
    pub const zero: @This() = .{ .x = 0, .y = 0, .z = 0 };
    x: i32,
    y: i32,
    z: i32,

    pub fn eql(self: @This(), other: @This()) bool {
        return std.meta.eql(self, other);
    }
};

pub const Rect2D = extern struct {
    offset: Offset2D,
    extent: Extent2D,

    pub fn eql(self: @This(), other: @This()) bool {
        return std.meta.eql(self, other);
    }
};

pub const Viewport = extern struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,

    pub fn eql(self: @This(), other: @This()) bool {
        return std.meta.eql(self, other);
    }
};

pub const XYColor = struct {
    x: f32,
    y: f32,
};

pub const HdrMetadata = struct {
    display_primary_red: XYColor,
    display_primary_green: XYColor,
    display_primary_blue: XYColor,
    white_point: XYColor,
    max_luminance: f32,
    min_luminance: f32,
    max_content_light_level: f32,
    max_frame_average_light_level: f32,
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

    pub fn format(self: @This(), writer: *std.Io.Writer) std.Io.Writer.Error!void {
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
                    .depth_stencil_image => |format| .{ .depth_stencil_image = format },
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

pub const ColorSpace = enum(i32) {
    srgb_nonlinear = Backend.named_color_spaces.srgb_nonlinear,
    hdr10_st2084 = Backend.named_color_spaces.hdr10_st2084,
    bt2020_linear = Backend.named_color_spaces.bt2020_linear,
    hdr10_hlg = Backend.named_color_spaces.hdr10_hlg,
    extended_srgb_linear = Backend.named_color_spaces.extended_srgb_linear,
    extended_srgb_nonlinear = Backend.named_color_spaces.extended_srgb_nonlinear,
    _,

    pub inline fn fromBackendType(value: Backend.ColorSpace) @This() {
        comptime assert(@sizeOf(Backend.ColorSpace) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(value));
    }

    pub inline fn asBackendType(self: @This()) Backend.ColorSpace {
        comptime assert(@sizeOf(Backend.ColorSpace) == @sizeOf(@This()));
        return @enumFromInt(@intFromEnum(self));
    }

    pub fn asBackendSlice(self: []const @This()) []const Backend.ColorSpace {
        comptime assert(@sizeOf(@This()) == @sizeOf(Backend.ColorSpace));
        comptime assert(@alignOf(@This()) == @alignOf(Backend.ColorSpace));
        return @ptrCast(self);
    }
};

pub const SurfaceFormatQuery = struct {
    pub const Result = struct {
        color_space: ColorSpace,
        image_format: ImageFormat,
        userdata: u32,
    };

    /// The query's color space.
    color_space: ColorSpace,
    /// The query's formats, sorted by priority.
    image_formats: []const ImageFormat,
    /// Pass through. Useful for checking which query was selected after initialization.
    userdata: u32,

    /// sRGB colors, you must use the sRGB transfer function before writing. This is the preferred
    /// way to display high quality SDR content, and all realistic Windows devices will be able
    /// to satisfy this query.
    pub fn linearSrgb(userdata: u32) @This() {
        return .{
            .color_space = .srgb_nonlinear,
            .image_formats = &.{
                // Supported by all realistic Windows devices in the Vulkan hardware database.
                .b8g8r8a8_unorm,
                // Available as a fallback.
                .r8g8b8a8_unorm,
                // Rarely supported, but available as a fallback.
                .r8g8b8a8_snorm,
                // Rarely supported, but available as a fallback.
                .a8b8g8r8_unorm,
                // Rarely supported, but available as a fallback.
                .a8b8g8r8_snorm,
                // Rarely supported, but available as a fallback.
                .b8g8r8a8_snorm,
            },
            .userdata = userdata,
        };
    }

    /// Similar to linear sRGB, but applies the transfer function for you. May be faster on low end
    /// hardware, but often cannot be written by a compute shader on real hardware--regardless
    /// of what the spec claims. All realistic Windows devices will be able to satisfy this query.
    pub fn nonlinearSrgb(userdata: u32) @This() {
        return .{
            .color_space = .srgb_nonlinear,
            .image_formats = &.{
                // Supported by all realistic Windows deviecs in the Vulkan hardware database
                .b8g8r8a8_srgb,
                // Rarely supported, but available as a fallback.
                .a8b8g8r8_srgb,
                // Rarely supported, but available as a fallback. Fails to gamma correct on AMD
                // drivers at the time of writing (but won't be chosen since they support
                // `b8g8r8a8_srgb`.)
                .r8g8b8a8_srgb,
            },
            .userdata = userdata,
        };
    }

    /// The most commonly supported HDR standard. Is typically available if the display is in
    /// HDR mode, you should provide a fallback for when it's not available.
    pub fn hdr10(userdata: u32) @This() {
        return .{
            .color_space = .hdr10_st2084,
            .image_formats = &.{
                // Most common.
                .a2b10g10r10_unorm,
                // Less common, but available as a fallback.
                .a2r10g10b10_unorm,
                // Less common, but availalbe as a fallback.
                .r16g16b16a16_sfloat,
            },
            .userdata = userdata,
        };
    }

    /// Widely supported according to the Vulkan hardware database, but doesn't appear to have wide
    /// usage for reasons that are unclear to me. Use if you know what you're doing.
    pub fn linearSrgbExtended(userdata: u32) @This() {
        return .{
            .color_space = .extended_srgb_linear,
            .image_formats = &.{
                // Most common.
                .r16g16b16a16_sfloat,
                // Rare, but available as a fallback.
                .a2r10g10b10_unorm,
                // Rare, but available as a fallback.
                .a2b10g10r10_unorm,
                // Rare, but available as a fallback.
                .r16g16b16a16_unorm,
            },
            .userdata = userdata,
        };
    }

    /// Widely supported according to the Vulkan hardware database, but doesn't appear to have wide
    /// usage for reasons that are unclear to me. Use if you know what you're doing.
    pub fn nonlinearSrgbExtended(userdata: u32) @This() {
        return .{
            .color_space = .extended_srgb_nonlinear,
            .image_formats = &.{
                // Most common.
                .r16g16b16a16_sfloat,
                // Rare, but available as a fallback.
                .a2r10g10b10_unorm,
                // Rare, but available as a fallback.
                .a2b10g10r10_unorm,
                // Rare, but available as a fallback.
                .r16g16b16a16_unorm,
            },
            .userdata = userdata,
        };
    }
};

/// Image formats.
///
/// Named in memory order. Note that DX12 formats are not necessarily named in memory order for
/// whatever reason. It's sometimes possible to discern the actual order through other bits and
/// pieces of the docs, sometimes it's ambiguous. I have not bothered to do this detective work, all
/// formats that don't have clear matches are just marked as ambiguous for now. It's possible that,
/// are we to add a DX12 backend, this will actually need more abstraction.
///
/// Good reference for support on DX12 level hardware:
/// - https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/hardware-support-for-direct3d-12-1-formats
///
/// For Vulkan conformance:
/// - https://docs.vulkan.org/spec/latest/chapters/formats.html
pub const ImageFormat = enum(i32) {
    undefined = Backend.named_image_formats.undefined,

    /// DX12 requires support for use as a storage image, and as a sampled image.
    r8_unorm = Backend.named_image_formats.r8_unorm,
    /// DX12 requires support for use as a write only storage image, and as a sampled image.
    r8_snorm = Backend.named_image_formats.r8_snorm,
    /// DX12 requires support for use as a storage image, and as a sampled image.
    r8_uint = Backend.named_image_formats.r8_uint,
    /// DX12 requires support for use as a storage image, and as a sampled image.
    r8_sint = Backend.named_image_formats.r8_sint,

    /// DX12 requires support for use as a storage image, and as a sampled image.
    r8g8b8a8_unorm = Backend.named_image_formats.r8g8b8a8_unorm,
    /// DX12 requires support for use as a write only storage image, and as a sampled image.
    r8g8b8a8_snorm = Backend.named_image_formats.r8g8b8a8_snorm,
    /// DX12 requires support for use as a write only storage image, and as a sampled image.
    r8g8b8a8_uint = Backend.named_image_formats.r8g8b8a8_uint,
    /// DX12 requires support for use as a write only storage image, and as a sampled image.
    r8g8b8a8_sint = Backend.named_image_formats.r8g8b8a8_sint,
    /// DX12 requires support for use as a sampled image.
    r8g8b8a8_srgb = Backend.named_image_formats.r8g8b8a8_srgb,

    /// DX12 requires support for use as a sampled image.
    b8g8r8a8_unorm = Backend.named_image_formats.b8g8r8a8_unorm,
    /// DX12 requires support for use as a sampled image.
    b8g8r8a8_srgb = Backend.named_image_formats.b8g8r8a8_srgb,

    /// DX12 requires support for use as a depth stencil target, and as a 1D or 2D sampled image.
    d24_unorm_s8_uint = Backend.named_image_formats.d24_unorm_s8_uint,
    /// DX12 requires support for use as a depth stencil target, and as a 1D or 2D sampled image.
    d32_sfloat = Backend.named_image_formats.d32_sfloat,

    /// DX12 requires support for use as a storage image, and as a sampled image.
    r16g16b16a16_sfloat = Backend.named_image_formats.r16g16b16a16_sfloat,
    /// DX12 requires support for use as a write only storage image, and as a sampled image.
    r16g16b16a16_unorm = Backend.named_image_formats.r16g16b16a16_unorm,
    /// DX12 requires support for use as a sampled image.
    r16g16b16a16_snorm = Backend.named_image_formats.r16g16b16a16_snorm,

    /// DX12 requires support for use as a sampled image.
    b5g6r5_unorm = Backend.named_image_formats.b5g6r5_unorm,
    /// DX12 requires support for use as a sampled image.
    b5g5r5a1_unorm = Backend.named_image_formats.b5g5r5a1_unorm,

    /// DX12 channel order is ambiguous, see enum docs.
    a8b8g8r8_srgb = Backend.named_image_formats.a8b8g8r8_srgb,
    /// DX12 channel order is ambiguous, see enum docs.
    a8b8g8r8_unorm = Backend.named_image_formats.a8b8g8r8_unorm,
    /// DX12 channel order is ambiguous, see enum docs.
    a8b8g8r8_snorm = Backend.named_image_formats.a8b8g8r8_snorm,
    /// DX12 channel order is ambiguous, see enum docs.
    b8g8r8a8_snorm = Backend.named_image_formats.b8g8r8a8_snorm,
    /// DX12 channel order is ambiguous, see enum docs.
    a2b10g10r10_unorm = Backend.named_image_formats.a2b10g10r10_unorm,
    /// DX12 channel order is ambiguous, see enum docs.
    a2r10g10b10_unorm = Backend.named_image_formats.a2r10g10b10_unorm,
    /// DX12 channel order is ambiguous, see enum docs.
    b10g11r11_ufloat = Backend.named_image_formats.b10g11r11_ufloat,
    /// DX12 channel order is ambiguous, see enum docs.
    r5g6b5_unorm = Backend.named_image_formats.r5g6b5_unorm,
    /// DX12 channel order is ambiguous, see enum docs.
    a1r5g5b5_unorm = Backend.named_image_formats.a1r5g5b5_unorm,
    /// DX12 channel order is ambiguous, see enum docs.
    r4g4b4a4_unorm = Backend.named_image_formats.r4g4b4a4_unorm,
    /// DX12 channel order is ambiguous, see enum docs.
    b4g4r4a4_unorm = Backend.named_image_formats.b4g4r4a4_unorm,
    /// DX12 channel order is ambiguous, see enum docs.
    r5g5b5a1_unorm = Backend.named_image_formats.r5g5b5a1_unorm,

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

/// Various levels of MSAA.
///
/// For the purposes of these docs, `unorm` formats are *not* considered integer formats, and it's
/// assumed that the image's only usage flag is `color_attachment` or `depth_stencil_attachment`--
/// adding other usage flags may conflict with MSAA support.
///
/// Querying for levels of MSAA support is not currently supported. This is a slightly involved
/// process as there are technically different results for various image types.
pub const Samples = enum {
    /// No MSAA, always supported.
    @"1",
    /// Two samples, not always supported.
    @"2",
    /// Four samples, supported on all realistic Windows hardware for non-integer formats at the
    /// time of writing
    @"4",
    /// Eight samples, supported on all realistic Windows hardware for non-integer formats at the
    /// time of writing.
    @"8",
    /// Sixteen samples, not always supported.
    @"16",
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
                    .depth_stencil => options.aspect,
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
                input_attachment: bool = false,
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

/// All shader stages.
pub const ShaderStage = enum {
    vertex,
    fragment,
    compute,
};

/// Graphics pipeline shader stages.
pub const GraphicsShaderStage = enum {
    vertex,
    fragment,
};

pub const ShaderStages = EnumBitSet(ShaderStage);

pub const BarrierStages = packed struct {
    top_of_pipe: bool = false,
    vertex: bool = false,
    fragment: bool = false,
    early_fragment_tests: bool = false,
    late_fragment_tests: bool = false,
    color_attachment_output: bool = false,
    compute: bool = false,
    copy: bool = false,
    blit: bool = false,
    bottom_of_pipe: bool = false,
    all_commands: bool = false,
};

fn EnumBitSet(T: type) type {
    const fields = @typeInfo(T).@"enum".fields;
    var struct_fields: [fields.len]std.builtin.Type.StructField = undefined;
    for (&struct_fields, fields) |*struct_field, enum_field| {
        struct_field.* = .{
            .name = enum_field.name,
            .type = bool,
            .default_value_ptr = @as(?*const anyopaque, @ptrCast(&false)),
            .is_comptime = false,
            .alignment = 0,
        };
    }
    return @Type(.{ .@"struct" = .{
        .layout = .@"packed",
        .fields = &struct_fields,
        .decls = &.{},
        .is_tuple = false,
    } });
}

fn nonZero(T: type, self: T) bool {
    const Int = std.meta.Int(.unsigned, @bitSizeOf(@TypeOf(self)));
    const int: Int = @bitCast(self);
    return int != 0;
}

pub const BindPoint = enum {
    graphics,
    compute,
};

pub const BindPoints = EnumBitSet(BindPoint);

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

    pub const InitGraphicsCmd = struct {
        pub const Stages = std.enums.EnumFieldStruct(GraphicsShaderStage, ShaderModule, null);

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

        pub const ColorComponents = packed struct {
            r: bool = false,
            g: bool = false,
            b: bool = false,
            a: bool = false,

            pub const all: @This() = .{ .r = true, .g = true, .b = true, .a = true };
        };

        pub const AttachmentBlendState = struct {
            pub const Factor = enum {
                zero,
                one,
                src_color,
                one_minus_src_color,
                dst_color,
                one_minus_dst_color,
                src_alpha,
                one_minus_src_alpha,
                dst_alpha,
                one_minus_dst_alpha,
                constant_color,
                one_minus_constant_color,
                constant_alpha,
                one_minus_constant_alpha,
                src_alpha_saturate,
            };

            pub const Op = enum {
                add,
                subtract,
                reverse_subtract,
                min,
                max,
            };
            src_color_factor: Factor,
            dst_color_factor: Factor,
            color_op: Op,
            src_alpha_factor: Factor,
            dst_alpha_factor: Factor,
            alpha_op: Op,
        };

        pub const LogicOp = enum {
            clear,
            @"and",
            and_reverse,
            copy,
            and_inverted,
            no_op,
            xor,
            @"or",
            nor,
            equivalent,
            invert,
            or_reverse,
            copy_inverted,
            or_inverted,
            nand,
            set,
        };

        /// We don't currently support the depth bounds check, as I wasn't able to get the correct
        /// behavior out of it on my AMD/nixOS setup.
        pub const DepthState = struct {
            @"test": bool,
            write: bool,
            compare_op: CompareOp,
        };

        pub const StencilState = struct {
            pub const OpState = struct {
                pub const Op = enum {
                    keep,
                    zero,
                    replace,
                    increment_clamp,
                    decrement_clamp,
                    invert,
                    increment_wrap,
                    decrement_wrap,
                };

                fail_op: Op,
                pass_op: Op,
                depth_fail_op: Op,
                compare_op: CompareOp,
                compare_mask: u32,
                write_mask: u32,
                reference: u32,
            };

            front: OpState,
            back: OpState,
        };

        name: DebugName,
        layout: Layout,
        stages: Stages,
        result: *Pipeline,
        input_assembly: InputAssembly,
        color_attachment_formats: []const ImageFormat,
        depth_attachment_format: ImageFormat,
        stencil_attachment_format: ImageFormat,
        rasterization_samples: Samples,
        alpha_to_coverage: bool,
        color_write_mask: ColorComponents,
        blend_state: ?AttachmentBlendState,
        depth_state: DepthState,
        stencil_state: ?StencilState,
        logic_op: ?LogicOp,
        blend_constants: [4]f32,
    };

    pub fn initGraphics(
        gx: *Gx,
        cmds: []const InitGraphicsCmd,
    ) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();
        log.debug("Pipeline.initGraphics ({})", .{cmds.len});
        if (std.debug.runtime_safety) {
            for (cmds) |cmd| {
                // Support for up to 4 is guaranteed by our Vulkan version. Once we update to 1.4,
                // we can bump this to 8.
                assert(cmd.color_attachment_formats.len <= 4);
            }
        }
        Backend.pipelinesCreateGraphics(gx, cmds);
        log.debug("Pipeline.initGraphics: success", .{});
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
        log.debug("Pipeline.initCompute ({})", .{cmds.len});
        Backend.pipelinesCreateCompute(gx, cmds);
        log.debug("Pipeline.initCompute: success", .{});
    }

    pub fn deinit(self: @This(), gx: *Gx) void {
        Backend.pipelineDestroy(gx, self);
    }

    /// For the time being, descriptor set layouts are tied to pipeline layouts.
    ///
    /// This may be changed in the future, but in general I've found it simpler and typically more
    /// efficient to just buffer up arguments and get the index from push constants or the instance
    /// ID.
    ///
    /// The problem with complex use of descriptor sets for per draw data is that unless you're
    /// using push descriptors which only have 73% adoption on Windows at the time of writing, you
    /// have to allocate a new descriptor set for each variant you use during a frame. This results
    /// in a lot of complexity, and a lot of calls into the driver to rebind descriptors.
    ///
    /// Keep in mind that separate descriptor sets can still share the underlying buffers that
    /// contain their data.
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
                    sampler: void,
                    sampled_image: void,
                    storage_image: void,
                    uniform_buffer: struct {
                        // We only support sizes up to 2^14, as Vulkan implementations don't have to support
                        // uniform buffers larger than this:
                        // https://registry.khronos.org/vulkan/specs/1.3/html/chap33.html#limits-minmax
                        size: u14,
                    },
                    storage_buffer: void,
                };

                name: []const u8,
                kind: Desc.Kind,
                count: u32 = 1,
                stages: ShaderStages,
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

            /// A range of push constant data. All Windows devices at the time of writing support at
            /// least 128 bytes of push constant data.
            ///
            /// Some APIs (e.g. Vulkan) allow setting the offset and size of each push constant
            /// range separately, whereas others like DX12 only allow you to specify non-overlapping
            /// ranges. For compatibility with the later class of APIs, we only allow
            /// non-overlapping ranges here. On an API like DX12 this can be converted directly to
            /// register + offsets since each register is a fixed size.
            pub const PushConstantRange = struct {
                stages: ShaderStages,
                size: u32,
            };

            name: DebugName,
            descs: []const Desc = &.{},
            push_constant_ranges: []const PushConstantRange = &.{},

            pub fn binding(comptime self: *const @This(), comptime name: []const u8) u32 {
                const result = comptime for (self.descs, 0..) |desc, i| {
                    if (std.mem.eql(u8, desc.name, name)) {
                        break i;
                    }
                } else @compileError("no such binding " ++ name);
                return result;
            }
        };

        pub const InitOptions = struct {
            pub const ImmutableSamplers = struct {
                binding: u32,
                samplers: []const Sampler,
            };

            layout: Layout.Options,
            immutable_samplers: []const ImmutableSamplers,
        };

        pub fn init(gx: *Gx, options: InitOptions) Layout {
            // Check that we're under the minimum guaranteed push constant size
            const max_pc_bytes = 128; // `maxPushConstantsSize` will be raised to 256 in Vulkan 1.4
            const min_pc_alignment = 4; // Vulkan requires multiple of 4
            var pc_bytes: u32 = 0;
            for (options.layout.push_constant_ranges) |range| {
                // Vulkan doesn't support 0 length push constant ranges
                assert(range.size > 0);
                // we can't just align forward ourselves as that makes writing weird
                assert(range.size % min_pc_alignment == 0);
                // Vulkan requires at least one stage to be set
                assert(nonZero(ShaderStages, range.stages));
                // Increment the total size
                pc_bytes += range.size;
            }
            assert(pc_bytes <= max_pc_bytes);

            // Check that our immutable samplers line up with the layout
            if (std.debug.runtime_safety) {
                for (options.immutable_samplers, 0..) |a, i| {
                    for (options.immutable_samplers[i + 1 ..]) |b| {
                        assert(a.binding != b.binding);
                    }
                    const desc = options.layout.descs[a.binding];
                    assert(desc.kind == .sampler);
                    assert(a.samplers.len == desc.count);
                }
            }

            // Create the layout
            return Backend.pipelineLayoutCreate(gx, options);
        }

        pub fn deinit(self: @This(), gx: *Gx) void {
            Backend.pipelineLayoutDestroy(gx, self);
        }
    };
};

pub const ImageFilter = enum {
    nearest,
    linear,
};

pub const CompareOp = enum {
    never,
    lt,
    eql,
    lte,
    gt,
    ne,
    gte,
    always,
};

pub const Sampler = enum(u64) {
    _,

    pub const Options = struct {
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
        pub const BorderColor = enum {
            float_transparent_black,
            int_transparent_black,
            float_opaque_black,
            int_opaque_black,
            float_opaque_white,
            int_opaque_white,
        };

        mag_filter: ImageFilter,
        min_filter: ImageFilter,
        mipmap_mode: ImageFilter,
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

    /// A descriptor set update, can be submitted to `Gx.updateDescSets`.
    pub const Update = struct {
        pub const Value = union(enum) {
            pub const Tag: type = @typeInfo(@This()).@"union".tag_type.?;
            pub const CombinedImageSampler = struct {
                view: ImageView,
                sampler: Sampler,
            };

            /// Prefer immutable samplers when possible.
            sampler: Sampler,
            sampled_image: ImageView,
            storage_image: ImageView,
            uniform_buf: Buf(.{ .uniform = true }).View,
            storage_buf: Buf(.{ .storage = true }).View,
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
};

pub const Device = struct {
    /// https://registry.khronos.org/vulkan/specs/1.3/html/chap33.html#limits-minmax
    pub const StorageBufSize = u27;

    /// https://registry.khronos.org/vulkan/specs/1.3/html/chap33.html#limits-minmax
    pub const max_uniform_buf_offset_alignment = 256;
    /// https://registry.khronos.org/vulkan/specs/1.3/html/chap33.html#limits-minmax
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
    surface_format: SurfaceFormatQuery.Result,
};

pub const ImageBarrier = struct {
    pub const Range = struct {
        base_mip_level: u32,
        mip_levels: u32,
        base_array_layer: u32,
        array_layers: u32,
        aspect: ImageAspect,

        pub fn first(aspect: ImageAspect) @This() {
            return .{
                .base_mip_level = 0,
                .mip_levels = 1,
                .base_array_layer = 0,
                .array_layers = 1,
                .aspect = aspect,
            };
        }
    };

    const End = struct {
        stages: BarrierStages,
        access: Access,
        layout: Layout,
    };

    pub const Layout = enum {
        undefined,
        general,
        read_only,
        attachment,
        transfer_src,
        transfer_dst,
    };

    image: ImageHandle,
    range: Range,
    src: End,
    dst: End,
};

pub const Access = packed struct {
    shader_read: bool = false,
    shader_write: bool = false,
    transfer_read: bool = false,
    transfer_write: bool = false,
    color_attachment_read: bool = false,
    color_attachment_write: bool = false,
    depth_stencil_attachment_read: bool = false,
    depth_stencil_attachment_write: bool = false,
    memory_read: bool = false,
    memory_write: bool = false,
};

pub const BufBarrier = struct {
    src_stages: BarrierStages,
    src_access: Access,
    dst_stages: BarrierStages,
    dst_access: Access,
    handle: BufHandle(.{}),
};

pub const ImageUpload = struct {
    pub const Region = struct {
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

    pub const Offset = struct { x: i32, y: i32, z: i32 };

    dst: ImageHandle,
    src: BufHandle(.{ .transfer_src = true }),
    base_mip_level: u32,
    mip_levels: u32,
    regions: []const Region,
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
        pub const Attachment = struct {
            view: ImageView,
            load_op: LoadOp,
            store_op: StoreOp,
            resolve_view: ?ImageView = null,
            resolve_mode: ResolveMode = .none,
        };

        pub const ResolveMode = enum {
            none,
            sample_zero,
            average,
            min,
            max,
        };

        const LoadOp = union(enum) {
            load: void,
            clear_color: [4]f32,
            clear_depth_stencil: struct { depth: f32, stencil: u32 },
            dont_care: void,
        };

        const StoreOp = enum {
            store,
            dont_care,
            none,
        };

        color_attachments: []const Attachment = &.{},
        depth_attachment: ?Attachment = null,
        stencil_attachment: ?Attachment = null,
        area: Rect2D,
        viewport: ?Viewport,
        scissor: ?Rect2D,
    };

    pub fn beginRendering(self: @This(), gx: *Gx, options: BeginRenderingOptions) void {
        Backend.cmdBufBeginRendering(gx, self, options);
        if (options.viewport) |viewport| Backend.cmdBufSetViewport(gx, self, viewport);
        if (options.scissor) |scissor| Backend.cmdBufSetScissor(gx, self, scissor);
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

    pub fn end(self: @This(), gx: *Gx) void {
        const zone = Zone.begin(.{ .src = @src() });
        defer zone.end();
        assert(gx.in_frame);
        Backend.cmdBufEnd(gx, self);
    }

    pub const BindPipelineOptions = struct {
        bind_point: BindPoint,
        pipeline: Pipeline,
    };

    pub fn bindPipeline(
        self: @This(),
        gx: *Gx,
        options: BindPipelineOptions,
    ) void {
        const zone = Zone.begin(.{ .src = @src() });
        defer zone.end();
        Backend.cmdBufBindPipeline(gx, self, options);
    }

    pub const BindDescSetOptions = struct {
        bind_points: BindPoints,
        layout: Pipeline.Layout.Handle,
        set: DescSet,
    };

    pub fn bindDescSet(self: @This(), gx: *Gx, options: BindDescSetOptions) void {
        const zone = Zone.begin(.{ .src = @src() });
        defer zone.end();
        Backend.cmdBufBindDescSet(gx, self, options);
    }

    pub const PushConstantSliceOptions = struct {
        pipeline_layout: Pipeline.Layout.Handle,
        stages: ShaderStages,
        offset: u32 = 0,
        data: []const u32,
    };

    pub fn pushConstantSlice(self: @This(), gx: *Gx, options: PushConstantSliceOptions) void {
        Backend.cmdBufPushConstants(gx, self, options);
    }

    pub fn pushConstantField(
        self: @This(),
        T: type,
        comptime field_name: []const u8,
        gx: *Gx,
        options: PushConstantOptions(@FieldType(T, field_name)),
    ) void {
        _ = extern struct { extern_type: @TypeOf(options.data.*) };
        Backend.cmdBufPushConstants(gx, self, .{
            .pipeline_layout = options.pipeline_layout,
            .stages = options.stages,
            .offset = options.offset + @offsetOf(T, field_name),
            .data = @ptrCast(std.mem.asBytes(options.data)),
        });
    }

    pub fn PushConstantOptions(T: type) type {
        return struct {
            pipeline_layout: Pipeline.Layout.Handle,
            stages: ShaderStages,
            offset: u32 = 0,
            data: *const T,
        };
    }

    pub fn pushConstant(
        self: @This(),
        T: type,
        gx: *Gx,
        options: PushConstantOptions(T),
    ) void {
        _ = extern struct { extern_type: @TypeOf(options.data.*) };
        Backend.cmdBufPushConstants(gx, self, .{
            .pipeline_layout = options.pipeline_layout,
            .stages = options.stages,
            .offset = options.offset,
            .data = @ptrCast(std.mem.asBytes(options.data)),
        });
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

    /// At the time of writing all Windows hardware on vulkan.gpu.info.org supports at least 256
    /// total work groups (x * y * z) at the time of writing, with the exception of some virtual
    /// devices that aren't relevant. The individual dimensions also support up to 256, with the
    /// exception of the z axis which often only supports up to 64.
    ///
    /// Details at the time of writing:
    /// - AMD subgroups have 32 or 64 threads
    /// - Nvidia subgroups have 32 threads
    /// - Intel varies, but appears to always be a power of two <= 32
    pub fn dispatch(self: @This(), gx: *Gx, groups: Extent3D) void {
        const zone = Zone.begin(.{ .src = @src() });
        defer zone.end();
        Backend.cmdBufDispatch(gx, self, groups);
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
            src_offset: u64 = 0,
            dst_offset: u64 = 0,
            size: u64,
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

    pub const BlitOptions = struct {
        pub const Subresource = struct {
            mip_level: u32,
            base_array_layer: u32,
            array_layers: u32,
            volume: Volume,
        };

        pub const Region = struct {
            src: Subresource,
            dst: Subresource,
            aspect: ImageAspect,
        };

        src: ImageHandle,
        dst: ImageHandle,
        regions: []const Region,
        filter: ImageFilter,
    };

    pub fn blit(self: @This(), gx: *Gx, options: BlitOptions) void {
        const zone = Zone.begin(.{ .src = @src() });
        defer zone.end();
        Backend.cmdBufBlit(gx, self, options);
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

test {
    _ = ext;
}
