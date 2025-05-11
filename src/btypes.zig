//! Types used by the backend interface.

const Ctx = @import("Ctx.zig");

pub const NamedImageFormats = struct {
    undefined: i32,
    r8g8b8a8_srgb: i32,
    d24_unorm_s8_uint: i32,
};

pub const ImageCreateResult = struct {
    handle: Ctx.ImageHandle,
    view: Ctx.ImageView,
    dedicated_memory: ?Ctx.Memory(.any),
};

pub const ImageOptions = struct {
    flags: Ctx.ImageFlags,
    dimensions: Ctx.Dimensions,
    format: Ctx.ImageFormat,
    extent: Ctx.ImageExtent,
    samples: Ctx.Samples,
    usage: ImageUsage,
    aspect: Ctx.ImageAspect,
    mip_levels: u32,
    array_layers: u32,
};

pub const ImageUsage = packed struct {
    transfer_src: bool = false,
    transfer_dst: bool = false,
    sampled: bool = false,
    storage: bool = false,
    color_attachment: bool = false,
    depth_stencil_attachment: bool = false,
    input_attachment: bool = false,
};

pub const MemoryCreateOptions = struct {
    pub const Access = union(enum) {
        none: void,
        write: struct { prefer_device_local: bool },
        read: void,

        fn asAccess(self: @This()) Ctx.MemoryKind.Access {
            return switch (self) {
                .none => .none,
                .write => .write,
                .read => .read,
            };
        }
    };

    pub const Usage = union(enum) {
        color_image: void,
        depth_stencil_image: Ctx.ImageFormat,

        fn asUsage(self: @This()) Ctx.MemoryKind.Usage {
            return switch (self) {
                .color_image => .{ .image = .{
                    .format = .color,
                } },
                .depth_stencil_image => |image| .{ .image = .{
                    .format = .{ .depth_stencil = image.format },
                } },
            };
        }
    };

    name: Ctx.DebugName,
    usage: Usage,
    access: Access = .none,
    size: u64,
};
