//! See `RenderTargetPool`.

const std = @import("std");
const gpu = @import("../root.zig");

const log = std.log.scoped(.gpu);

const Allocator = std.mem.Allocator;
const Gx = gpu.Gx;
const ImageKind = gpu.ImageKind;
const Image = gpu.Image;
const DebugName = gpu.DebugName;
const ImageBumpAllocator = gpu.ext.ImageBumpAllocator;

/// A pool for managing render targets.
///
/// A render target is just an image. However, the size of a render target typically depends on the
/// surface size, which on PC typically depends on the window size. This may be a direct
/// relationship (e.g. a full resolution color buffer), or an indirect one (e.g. a quarter
/// resolution blur buffer, or a game rendered at half resolution.)
///
/// In practice this becomes difficult to manage: on PC the window size can change at runtime, which
/// necessitates destroying and recreating all render targets to resize them.
///
/// This abstraction allows specifying render target images according to a virtual coordinate
/// system, returns persistent handles to the created images through a layer of indirection, and
/// then manages recreating of the render targets for you.
///
/// For example, you may set up the pool to have a virtual extent of 1920x1080 and a physical extent
/// of 1920x1080. Under this setup, a 1920x1080 is surface sized, a 960x540 image is half surface
/// sized. Virtual coordinates were chosen over floats ranging from 0 to 1 as this avoids needing to
/// create entirely new option structs for image creation.
pub fn RenderTargetPool(kind: ImageKind) type {
    return struct {
        const Pool = @This();

        /// A persistent render target handle. Store this frame to frame instead of storing the
        /// actual images which are transient.
        pub const Handle = enum(u32) {
            _,

            /// Internal helper for initializing a render target.
            fn init(self: @This(), pool: *Pool, gx: *Gx) void {
                // Initialize the image
                var info: ImageBumpAllocator(kind).AllocOptions = pool.info.items[@intFromEnum(self)];
                info.image.extent = self.extent(pool);
                const image = pool.allocator.alloc(gx, info);
                pool.images.items[@intFromEnum(self)] = image;
            }

            /// Returns the extent of this handle.
            fn extent(self: @This(), pool: *const Pool) gpu.ImageExtent {
                const info: ImageBumpAllocator(kind).AllocOptions = pool.info.items[@intFromEnum(self)];
                const x_scale: f32 = @as(f32, @floatFromInt(pool.physical_extent.width)) / @as(f32, @floatFromInt(pool.virtual_extent.width));
                const y_scale: f32 = @as(f32, @floatFromInt(pool.physical_extent.height)) / @as(f32, @floatFromInt(pool.virtual_extent.height));
                return .{
                    .width = @intFromFloat(x_scale * @as(f32, @floatFromInt(info.image.extent.width))),
                    .height = @intFromFloat(y_scale * @as(f32, @floatFromInt(info.image.extent.height))),
                    .depth = info.image.extent.depth,
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

        name: [:0]const u8,
        physical_extent: gpu.Extent2D,
        virtual_extent: gpu.Extent2D,
        info: std.ArrayListUnmanaged(ImageBumpAllocator(kind).AllocOptions),
        images: std.ArrayListUnmanaged(gpu.Image(kind)),
        allocator: ImageBumpAllocator(kind),
        desc_sets: [gpu.global_options.max_frames_in_flight]gpu.DescSet,
        storage_binding: ?u32,
        sampled_binding: ?u32,

        /// Options for `init`.
        pub const Options = struct {
            virtual_extent: gpu.Extent2D,
            physical_extent: gpu.Extent2D,
            capacity: u8,
            allocator: ImageBumpAllocator(kind).Options,
            desc_sets: [gpu.global_options.max_frames_in_flight]gpu.DescSet,
            storage_binding: ?u32,
            sampled_binding: ?u32,
        };

        /// Initialize a render target pool.
        pub fn init(gpa: Allocator, gx: *Gx, options: Options) Allocator.Error!@This() {
            var allocator: ImageBumpAllocator(kind) = try .init(gpa, gx, options.allocator);
            errdefer allocator.deinit(gpa, gx);

            log.debug("Initializing render target pool '{s}' with physical extent {}x{}", .{
                options.allocator.name,
                options.physical_extent.width,
                options.physical_extent.height,
            });

            var images: std.ArrayListUnmanaged(Image(kind)) = try .initCapacity(
                gpa,
                options.capacity,
            );
            errdefer images.deinit(gpa);

            const info: std.ArrayListUnmanaged(ImageBumpAllocator(kind).AllocOptions) = try .initCapacity(
                gpa,
                options.capacity,
            );
            errdefer info.deinit(gpa);

            return .{
                .name = options.allocator.name,
                .physical_extent = options.physical_extent,
                .virtual_extent = options.virtual_extent,
                .images = images,
                .info = info,
                .allocator = allocator,
                .desc_sets = options.desc_sets,
                .storage_binding = options.storage_binding,
                .sampled_binding = options.sampled_binding,
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
            options: ImageBumpAllocator(kind).AllocOptions,
        ) Handle {
            if (self.images.items.len == self.images.capacity) @panic("OOB");
            const handle: Handle = @enumFromInt(self.images.items.len);
            self.info.appendAssumeCapacity(options);
            _ = self.images.addOneAssumeCapacity();
            handle.init(self, gx);
            return handle;
        }

        /// Updates the physical extent, recreating all render targets if needed. You must ensure
        /// that the render targets are not in use before calling this, the recommended approach is
        /// to use `waitIdle` since recreating render targets may take a moment anyway.
        ///
        /// See also `suboptimal`.
        pub fn recreate(
            self: *@This(),
            gx: *Gx,
            physical_extent: gpu.Extent2D,
        ) void {
            log.info("Recreating render target pool '{s}' with physical extent {}x{}", .{
                self.name,
                physical_extent.width,
                physical_extent.height,
            });

            // Update the physical extent
            self.physical_extent = physical_extent;

            // Destroy the existing render targets
            for (self.images.items) |image| {
                image.deinit(gx);
            }
            self.allocator.reset(gx);

            // Recreate the render targets
            for (0..self.images.items.len) |i| {
                const handle: Handle = @enumFromInt(i);
                handle.init(self, gx);
            }
        }

        /// Returns true if you should recreate the render targets for the best resizing experience.
        /// This is not required for correctness, it is a convenient way to balance keeping the
        /// render targets sized relative to the surface with maintaining a smooth resize
        /// experience. Recommended usage is to call at the end of your frame after presentation
        /// since this minimizes the latency introduced when recreate is necessary.
        pub fn suboptimal(
            self: *@This(),
            /// This should be a timer that's reset every time the surface is resized.
            resize_timer: *std.time.Timer,
            surface_extent: gpu.Extent2D,
        ) bool {
            // Early out if the extent hasn't changed.
            if (self.physical_extent.eql(surface_extent)) return false;

            // Early out if the extent is 0 sized.
            if (surface_extent.width == 0 or surface_extent.height == 0) return false;

            // Early out if the window was recently resized since we may be still be in an active
            // resize and should let it complete, unless the new size is dramatically larger than
            // the current one. We use a dedicated timer here rather than allow relying on the in
            // game delta time, since on some platforms (e.g. Windows) the OS takes over control
            // flow during resizes, which is likely to pause the in game delta time.
            const scale = @max(
                surface_extent.width / self.physical_extent.width,
                surface_extent.height / self.physical_extent.height,
            );
            const needs_recreate = scale > 8 or resize_timer.read() > 100000000;
            if (!needs_recreate) return false;

            // The render target pool should be recreated. We don't do the actual recreate here
            // since we need to wait for the GPU to idle first, and there may be multiple render
            // target pools (e.g. one for depth) so we don't want to do the wait internally and end
            // up with a redundant wait.
            return true;
        }
    };
}
