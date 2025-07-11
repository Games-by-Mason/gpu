//! A color image upload queue.
//!
//! Operates on a fixed size staging buffer, writes to the memory and command buffer provided by the
//! caller.
//!
//! To upload an image, call `beginWrite`, and then write your data to the provided writer. Repeat
//! for each image. To initiate the upload, call submit on the command buffer you provided.
//!
//! You may not use the staging buffer view until the work completes asynchronously on the GPU, see
//! *Uploading Asynchronously* for the recommended pattern.
//!
//! Caller is responsible for issuing the barriers to transition the image from `transfer_dst` once
//! they wish to begin using them.
//!
//! # Uploading Synchronously
//!
//! If you only need to upload a small amount of data, you can probably load your images
//! synchronously to a single `ImageUploadQueue`. You can't delete the staging buffer until the
//! upload is done, using a delete queue is recommended.
//!
//! Attempting to calculate the exact size of the staging buffer or image memory isn't recommended
//! as it limits your flexibility. When working with a small amount of data, this shouldn't be a
//! concern, and when working with a large amount of data you typically upload asynchronously which
//! means you're reusing staging buffers anyway.
//!
//! # Uploading Asynchronously & Repeated Uploads
//!
//! If you're uploading a large amount of data, you probably should do it asynchronously.
//!
//! Eventually `gpu` will get support for multiple queues
//! (https://github.com/Games-by-Mason/gpu/issues/1), but even still you likely want to
//! support hardware that only provides a single queue. This means that you need to divide up your
//! work into frame sized chunks.
//!
//! The recommended approach is to create one staging buffer view per frame and flight and keep them
//! live. Then at the start of each frame, you can create an `ImageUploadQueue` with the current
//! frame's staging view. Assets should be loaded from disk on a background thread or threads.
//! The main thread polls to see if the background threads have completed any work, and if they
//! have, checks if there's space in the current upload queue. If there is, it writes them.
//!
//! By tuning the size of the staging buffer, you can limit how much work is done one any single
//! frame and provide a smooth experience. Keep in mind that the bottleneck is likely the read from
//! disk, not the GPU upload, and you can set your values accordingly. Setting the buffer too small
//! is unlikely to affect performance much, but it needs to be at least as large as the largest
//! image.
//!
//! This approach may seem sub-optimal since it involves an extra staging buffer on the CPU in the
//! background thread. In practice, this buffer is typically necessary anyway since images often
//! need to be decompressed or unpacked from asset bundles before being uploaded. That work can be
//! written directly to the provided writer. Even if you're loading data that needs no processing,
//! increasing the pipeline length for asynchronous work shouldn't be a concern, and when doing
//! synchronous uploads you can pipe your file directly to the writer.

const gpu = @import("../root.zig");
const ImageBumpAllocator = gpu.ext.ImageBumpAllocator;

const Gx = gpu.Gx;
const CmdBuf = gpu.CmdBuf;
const Writer = gpu.Writer;
const UploadBuf = gpu.UploadBuf;
const DebugName = gpu.DebugName;
const Image = gpu.Image;

staging: gpu.BufHandle(.{ .transfer_src = true }),
writer: Writer,

/// Creates an image upload queue. If uploading over the course of multiple frames, you should
/// create one per frame in flight and reuse them.
pub fn init(staging: gpu.UploadBuf(.{ .transfer_src = true }).View) @This() {
    return .{
        .staging = staging.handle,
        .writer = staging.writer(),
    };
}

/// Call this before writing the data for an image to `writer`.
///
/// Panics on staging buffer overflow. If you want to handle this case, align the writer to
/// `gpu.buffer_copy_offset_alignment` before calling and then check if there's enough space left.
pub fn beginWrite(
    self: *@This(),
    gx: *Gx,
    cb: CmdBuf,
    allocator: *ImageBumpAllocator(.color),
    options: ImageBumpAllocator(.color).AllocOptions,
) gpu.Image(.color) {
    // This alignment is required by DX12. With Vulkan it's device dependent and optional, in
    // practice real GPUs may have the value set to 1. If DX12 ever lifts this requirements we could
    // elide this padding, though keep in mind that block based formats would still need to be
    // aligned to their blocks or such which is happening implicitly here.
    self.writer.alignForward(gpu.Device.buffer_copy_offset_alignment);

    // Create the image.
    const image: gpu.Image(.color) = allocator.alloc(gx, options);

    // Issue the image transitions and image uploads. Theoretically it's faster to batch all of
    // these image transitions in a single call to `cb.barrier`. This requires queuing them up CPU
    // side, which increases complexity. In practice, even when uploading 1000s of images, I was not
    // able to measure a difference in performance, the bottleneck is clearly the actual copy so
    // we're opting to keep thing simple.
    cb.barriers(gx, .{ .image = &.{
        .undefinedToTransferDst(.{
            .handle = image.handle,
            .src_stages = .{ .top_of_pipe = true },
            .range = .first,
            .aspect = .{ .color = true },
        }),
    } });

    cb.uploadImage(gx, .{
        .dst = image.handle,
        .src = self.staging,
        .base_mip_level = 0,
        .mip_levels = 1,
        .regions = &.{
            .init(.{
                .aspect = .{ .color = true },
                .image_extent = options.image.extent,
                .buffer_offset = self.writer.pos,
            }),
        },
    });

    // Return the image to the user.
    return image;
}
