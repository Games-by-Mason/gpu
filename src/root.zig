//! A light graphics API abstraction.
//!
//! See the README for more information.

const std = @import("std");
const root = @import("root");

pub const tracy = @import("tracy");

/// Compile time options for the library. You must declare a constant of this type in your root file
/// named `gpu_options` to configure the library.
pub const Options = struct {
    /// Must implement `IBackend`
    Backend: type,
    /// Pipeline layouts can be created at runtime, but specifying them at compile time offers
    /// additional type safety.
    combined_pipeline_layouts: []const *const Ctx.CombinedPipelineLayout.InitOptions = &.{},
    max_frames_in_flight: u8 = 2,
    max_queues: struct {
        graphics: u6 = 1,
        compute: u6 = 0,
        transfer: u6 = 1,
    },
    tracy_query_pool_capacity: u16 = 256,
    max_command_buffers_per_frame: u8,
    blocking_zone_color: tracy.Color = .dark_sea_green4,
};

pub const max_queues = @as(u8, @intCast(options.max_queues.graphics)) +
    @as(u8, @intCast(options.max_queues.compute)) +
    @as(u8, @intCast(options.max_queues.transfer));

const options_name = "gpu_options";
pub const options: Options = if (@hasDecl(root, "gpu_options"))
b: {
    break :b root.gpu_options;
} else @compileError("root is missing " ++ options_name);

pub const Ctx = @import("Ctx.zig");
pub const writers = @import("writers.zig");

pub const IBackend = @import("ibackend.zig").IBackend;
