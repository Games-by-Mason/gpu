//! Optional extensions not part of the core library.

const std = @import("std");

const image_bump_allocator = @import("ext/image_bump_allocator.zig");

pub const ImageBumpAllocator = @import("ext/image_bump_allocator.zig").ImageBumpAllocator;
pub const DeleteQueue = @import("ext/DeleteQueue.zig");
pub const ImageUploadQueue = @import("ext/ImageUploadQueue.zig");
pub const ModTimer = @import("ext/mod_timer.zig").ModTimer;
pub const RenderTarget = @import("ext/render_target.zig").RenderTarget;
pub const ScopedArena = @import("ext/ScopedArena.zig");
pub const bufPart = @import("ext/buf_part.zig").bufPart;
pub const colors = @import("ext/colors.zig");
pub const gaussian = @import("ext/gaussian.zig");

test {
    _ = colors;
    _ = gaussian;
}
