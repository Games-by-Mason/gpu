//! See `RenderTargetPool`.

const std = @import("std");
const gpu = @import("../root.zig");

const Allocator = std.mem.Allocator;
const Gx = gpu.Gx;
const ImageKind = gpu.ImageKind;
const Image = gpu.Image;
const DebugName = gpu.DebugName;
const ImageBumpAllocator = gpu.ext.ImageBumpAllocator;
const DescSet = gpu.DescSet;

/// A pool for managing render targets.
///
/// A render target is just an image. However, the size of a render target typically depends on the
/// window size. Most render targets are sized the same as the window, some may be a fraction of
/// the window size (e.g. a blur) or at a multiple of the window size (e.g. super sampling.)
///
/// In practice this becomes difficult to manage: the window size can change at runtime, which
/// necessitates destroying and recreating all render targets to resize them. Furthermore, if the
/// image is recreated, it will need to be updated in any descriptor sets its currently bound to.
///
/// This abstraction allows specifying render target images according to a virtual coordinate
/// system, and then manages recreating and rebinding the render target images for you.
///
/// For example, you may set up the pool to have a virtual extent of 1920x1080 and a physical extent
/// of 1920x1080. Under this, a 960x540 image will be half window sized. However, if the physical
/// extent later changes to 3840x2160, the image will also double in size. This will also emit the
/// required descriptor set update commands.
///
/// Updating the descriptor sets automatically is made possible by a bindless design. That is to
/// say, this abstraction assumes all your render targets are bound to a single descriptor set per
/// frame in flight, and indexed via the handle returned by this pool. It's up to you how to decide
/// in the shader which index to access, but the intended approach is that you'd index an arguments
/// array based on the base instance or invocation ID.
///
/// If you need to represent multiple access patterns or formats in the shader, you can alias the
/// uniform under multiple types.
pub fn RenderTargetPool(kind: ImageKind) type {
    return struct {
        const Pool = @This();

        /// A persistent render target handle.
        pub const Handle = enum(u8) {
            _,

            /// Internal helper for initializing a render target.
            fn init(
                self: @This(),
                pool: *Pool,
                gx: *Gx,
                updates: *std.ArrayList(gpu.DescSet.Update),
            ) void {
                var desc: ImageBumpAllocator(kind).AllocOptions = pool.descs.items[@intFromEnum(self)];
                const image = pool.allocator.alloc(gx, desc);
                desc.image.extent = self.extent(pool);
                pool.images.items[@intFromEnum(self)] = image;
                for (pool.desc_sets) |desc_set| {
                    updates.append(.{
                        .set = desc_set,
                        .binding = pool.binding,
                        .value = .{
                            .storage_image = image.view,
                        },
                    }) catch @panic("OOB");
                }
            }

            /// Returns the extent of this handle.
            fn extent(self: @This(), pool: *const Pool) gpu.ImageExtent {
                const desc: ImageBumpAllocator(kind).AllocOptions = pool.descs.items[@intFromEnum(self)];
                const x_scale: f32 = @as(f32, @floatFromInt(pool.physical_extent.width)) / @as(f32, @floatFromInt(pool.virtual_extent.width));
                const y_scale: f32 = @as(f32, @floatFromInt(pool.physical_extent.height)) / @as(f32, @floatFromInt(pool.virtual_extent.height));
                return .{
                    .width = @intFromFloat(x_scale * @as(f32, @floatFromInt(desc.image.extent.width))),
                    .height = @intFromFloat(y_scale * @as(f32, @floatFromInt(desc.image.extent.height))),
                    .depth = desc.image.extent.depth,
                };
            }

            /// Gets a the image for this handle. This image will be invalidated on `recreate`, you
            /// shouldn't store or free it.
            pub fn get(self: @This(), pool: *const Pool) gpu.Image(kind) {
                return pool.images.items[@intFromEnum(self)];
            }

            /// Gets a sized image view for this handle. This view will be invalidated on
            /// `recreate`, you shouldn't store it.
            pub fn getSizedView(self: @This(), pool: *const Pool) gpu.ImageView.Sized2D {
                const scaled = self.extent(pool);
                return .{
                    .extent = .{
                        .width = scaled.width,
                        .height = scaled.height,
                    },
                    .view = self.get(pool).view,
                };
            }
        };

        physical_extent: gpu.Extent2D,
        virtual_extent: gpu.Extent2D,
        descs: std.ArrayListUnmanaged(ImageBumpAllocator(kind).AllocOptions),
        images: std.ArrayListUnmanaged(gpu.Image(kind)),
        allocator: ImageBumpAllocator(kind),
        desc_sets: [gpu.global_options.max_frames_in_flight]gpu.DescSet,
        binding: u32,

        /// Options for `init`.
        pub const Options = struct {
            virtual_extent: gpu.Extent2D,
            physical_extent: gpu.Extent2D,
            capacity: u8,
            allocator: ImageBumpAllocator(kind).Options,
            desc_sets: [gpu.global_options.max_frames_in_flight]gpu.DescSet,
            binding: u32,
        };

        /// Initialize a render target pool.
        pub fn init(gpa: Allocator, gx: *Gx, options: Options) Allocator.Error!@This() {
            var allocator: ImageBumpAllocator(kind) = try .init(gpa, gx, options.allocator);
            errdefer allocator.deinit(gpa, gx);

            var images: std.ArrayListUnmanaged(Image(kind)) = try .initCapacity(
                gpa,
                options.capacity,
            );
            errdefer images.deinit(gpa);

            const descs: std.ArrayListUnmanaged(ImageBumpAllocator(kind).AllocOptions) = try .initCapacity(
                gpa,
                options.capacity,
            );
            errdefer descs.deinit(gpa);

            return .{
                .physical_extent = options.physical_extent,
                .virtual_extent = options.virtual_extent,
                .images = images,
                .descs = descs,
                .allocator = allocator,
                .desc_sets = options.desc_sets,
                .binding = options.binding,
            };
        }

        /// Destroy the render target pool and all owned images.
        pub fn deinit(self: *@This(), gpa: Allocator, gx: *Gx) void {
            for (self.images.items) |image| {
                image.deinit(gx);
            }
            self.images.deinit(gpa);
            self.allocator.deinit(gpa, gx);
            self.* = undefined;
        }

        /// Allocates a render target. The width and height are automatically scaled according to
        /// the current virtual extent divided by the current physical extent.
        pub fn alloc(
            self: *@This(),
            gx: *Gx,
            updates: *std.ArrayList(DescSet.Update),
            options: ImageBumpAllocator(kind).AllocOptions,
        ) Handle {
            if (self.images.items.len == self.images.capacity) @panic("OOB");
            const handle: Handle = @enumFromInt(self.images.items.len);
            self.descs.appendAssumeCapacity(options);
            _ = self.images.addOneAssumeCapacity();
            handle.init(self, gx, updates);
            return handle;
        }

        /// Updates the physical extent, recreating all render targets if needed. You must ensure
        /// that the render targets are not in use before calling this. The recommended approach is
        /// to use `waitIdle`, recreating render targets is an allocation and as such may take a
        /// moment anyway.
        pub fn recreate(
            self: *@This(),
            gx: *Gx,
            physical_extent: gpu.Extent2D,
            updates: std.ArrayListUnmanaged(DescSet.Update),
        ) error.OutOfBounds!void {
            // Update the physical extent, or early out if it hasn't changed
            if (self.physical_extent == physical_extent) return;
            self.physical_extent = physical_extent;

            // Destroy the existing render targets
            for (self.images.items) |image| {
                image.deinit(gx);
            }
            self.allocator.reset();

            // Recreate the render targets
            for (0..self.images.len) |i| {
                const handle: Handle = @enumFromInt(i);
                handle.init(self, gx, updates);
            }
        }
    };
}
