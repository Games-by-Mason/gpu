//! A light graphics API abstraction.
//!
//! See the README for more information.

const std = @import("std");
const root = @import("root");

pub const tracy = @import("tracy");

/// Compile time options for the library. You must declare a constant of this type in your root file
/// named `gpu_options` to configure the library.
pub const Options = struct {
    Backend: type,
    max_frames_in_flight: u4 = 2,
    blocking_zone_color: tracy.Color = .dark_sea_green4,
    init_pipelines_buf_len: u32 = 16,
    init_desc_pool_buf_len: u32 = 16,
    update_desc_sets_buf_len: u32 = 16,
    combined_pipeline_layout_create_buf_len: u32 = 16,
};

pub const options: Options = root.gpu_options;

pub const Ctx = @import("Ctx.zig");
pub const writers = @import("writers.zig");

pub const btypes = @import("btypes.zig");
