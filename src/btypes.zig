//! Types used by the backend interface.

const Gx = @import("Gx.zig");
const gpu = @import("root.zig");
const Backend = gpu.global_options.Backend;

pub const BackendInitResult = struct {
    backend: Backend,
    device: gpu.Device,
};

pub const NamedImageFormats = struct {
    undefined: i32,

    r8g8b8a8_unorm: i32,
    r8g8b8a8_snorm: i32,
    r8g8b8a8_uscaled: i32,
    r8g8b8a8_sscaled: i32,
    r8g8b8a8_uint: i32,
    r8g8b8a8_sint: i32,

    r8g8b8a8_srgb: i32,

    d24_unorm_s8_uint: i32,
};

pub const ImageCreateResult = struct {
    handle: gpu.ImageHandle,
    view: gpu.ImageView,
    dedicated_memory: ?gpu.Memory(.any),
};

pub const ImageOptions = struct {
    flags: gpu.ImageFlags,
    dimensions: gpu.Dimensions,
    format: gpu.ImageFormat,
    extent: gpu.ImageExtent,
    samples: gpu.Samples,
    usage: ImageUsage,
    aspect: gpu.ImageAspect,
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

        fn asAccess(self: @This()) Gx.MemoryKind.Access {
            return switch (self) {
                .none => .none,
                .write => .write,
                .read => .read,
            };
        }
    };

    pub const Usage = union(enum) {
        color_image: void,
        depth_stencil_image: gpu.ImageFormat,

        fn asUsage(self: @This()) gpu.MemoryKind.Usage {
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

    name: gpu.DebugName,
    usage: Usage,
    access: Access = .none,
    size: u64,
};
