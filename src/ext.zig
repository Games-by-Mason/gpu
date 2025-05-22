//! Optional extensions not part of the core library.

const image_bump_allocator = @import("ext/image_bump_allocator.zig");

pub const ImageBumpAllocator = @import("ext/image_bump_allocator.zig").ImageBumpAllocator;
pub const DeleteQueue = @import("ext/delete_queue.zig").DeleteQueue;
