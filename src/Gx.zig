//! The graphics context.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.gpu);
const tracy = @import("tracy");
const Zone = tracy.Zone;
const gpu = @import("root.zig");
const builtin = @import("builtin");
const btypes = @import("btypes.zig");

const Extent2D = gpu.Extent2D;
const Buf = gpu.Buf;
const ImageView = gpu.ImageView;
const Sampler = gpu.Sampler;
const DescSet = gpu.DescSet;
const Device = gpu.Device;

const Ctx = @This();

const Backend = gpu.global_options.Backend;

/// Backend specific state.
backend: Backend,
/// Device information.
device: Device,
/// The number of frames that can be in flight at once.
frames_in_flight: u4,
/// The current frame in flight.
frame: u8 = 0,
/// Whether or not we're currently in between `beginFrame` and `endFrame`.
in_frame: bool = false,
/// Whether or not timestamp queries are enabled.
timestamp_queries: bool,
/// The live Tracy queries for each frame.
tracy_queries: [gpu.global_options.max_frames_in_flight]u16 = @splat(0),
/// Whether or not validation is enabled.
validate: bool,

/// Initialization options.
pub const Options = struct {
    pub const ColorSpace = enum {
        srgb,
        linear,
    };

    pub const DebugMode = enum(u8) {
        /// Enables graphics API validation and debug output. High performance cost.
        ///
        /// Will emit warning if not available on host.
        validate = 2,
        /// Enables debug output. Minimal to no performance cost, may aid profiling software in
        /// providing readable output.
        ///
        /// Will emit warning if not available on host.
        output = 1,
        /// No debugging support.
        none = 0,

        pub fn gte(lhs: @This(), rhs: @This()) bool {
            return @intFromEnum(lhs) >= @intFromEnum(rhs);
        }
    };

    /// The default device type ranking.
    pub const default_device_type_ranks = b: {
        var ranks = std.EnumArray(Device.Kind, u8).initFill(0);
        ranks.set(.discrete, 2);
        ranks.set(.integrated, 1);
        break :b ranks;
    };

    /// A semantic version.
    pub const Version = struct {
        major: u7,
        minor: u10,
        patch: u12,
    };

    /// The application name, the graphics API may expose this to drivers.
    app_name: ?[:0]const u8 = null,
    /// The application version, the graphics API may expose this to drivers.
    app_version: Version = .{
        .major = 0,
        .minor = 0,
        .patch = 0,
    },
    /// The engine name, the graphics API may expose this to drivers.
    engine_name: ?[:0]const u8,
    /// The engine version, the graphics API may expose this to drivers.
    engine_version: Version = .{
        .major = 0,
        .minor = 0,
        .patch = 0,
    },
    /// The number of frames in flight.
    frames_in_flight: u4,
    /// The device type rankings.
    device_type_ranks: std.EnumArray(Device.Kind, u8) = default_device_type_ranks,
    /// Whether or not to enable timestamp queries.
    timestamp_queries: bool,
    /// What level of debugging to enable.
    debug: DebugMode = if (builtin.mode == .Debug) .validate else .none,
    /// Disables potentially problematic features. For example, disables all implicit layers in
    /// Vulkan. This may disrupt functionality expected by the user and should only be enabled
    /// when a problem occurs.
    safe_mode: bool = false,
    /// Whether or not to force maximum alignment, may be useful for diagnosing some memory related
    /// issues.
    max_alignment: bool = false,
    /// The swapchain's color space.
    swapchain_color_space: ColorSpace,
    /// Backend specific options.
    backend: Backend.Options,
};

/// Initializes the graphics context.
pub fn init(gpa: Allocator, options: Options) @This() {
    const zone = tracy.Zone.begin(.{ .name = "gpu init", .src = @src() });
    defer zone.end();
    log.debug("Initializing GPU frontend", .{});

    assert(options.frames_in_flight > 0);
    assert(options.frames_in_flight <= gpu.global_options.max_frames_in_flight);

    const backend_result = Backend.init(gpa, options);

    var gx: @This() = .{
        .backend = backend_result.backend,
        .device = backend_result.device,
        .frames_in_flight = options.frames_in_flight,
        .timestamp_queries = options.timestamp_queries,
        .validate = options.debug.gte(.validate),
    };

    if (options.max_alignment) {
        gx.device.uniform_buf_offset_alignment = Device.max_uniform_buf_offset_alignment;
        gx.device.storage_buf_offset_alignment = Device.max_storage_buf_offset_alignment;
    }

    return gx;
}

/// Destroys the graphics context. Must not be in use, see `waitIdle`.
pub fn deinit(self: *@This(), gpa: Allocator) void {
    Backend.deinit(self, gpa);
    self.* = undefined;
}

/// Gets a pointer to the backend of the current type, or emits a compiler error if the type does
/// not match.
pub fn getBackend(self: *@This(), T: type) *T {
    assert(gpu.global_options.Backend == T);
    return &self.backend;
}

/// See `getBackend`.
pub fn getBackendConst(self: *const @This(), T: type) *const T {
    assert(gpu.global_options.Backend == T);
    return &self.backend;
}

/// Options for ending the current frame.
pub const EndFrameOptions = struct {
    /// If true, presents the current image, otherwise ends the frame without presentation.
    ///
    /// Some drivers sometimes block here waiting for the swapchain image instead of when acquiring
    /// it.
    present: bool = true,
};

/// Ends the current frame.
pub fn endFrame(self: *@This(), options: EndFrameOptions) void {
    const zone = Zone.begin(.{ .src = @src() });
    defer zone.end();
    const blocking_zone = Zone.begin(.{
        .src = @src(),
        .color = gpu.global_options.blocking_zone_color,
    });
    defer blocking_zone.end();
    Backend.endFrame(self, options);
    const Frame = @TypeOf(self.frame);
    const FramesInFlight = @TypeOf(self.frames_in_flight);
    comptime assert(std.math.maxInt(FramesInFlight) < std.math.maxInt(Frame));
    self.frame = (self.frame + 1) % self.frames_in_flight;
    assert(self.in_frame);
    self.in_frame = false;
}

/// Acquires the next swapchain image, blocking until it's available if necessary. Returns the
/// nanoseconds spent blocking.
///
/// Returns null if the swapchain needed to be recreated, in which case you should drop this frame.
pub fn acquireNextImage(self: *@This(), framebuf_extent: Ctx.Extent2D) ImageView.Sized2D {
    const zone = Zone.begin(.{
        .src = @src(),
        .color = gpu.global_options.blocking_zone_color,
    });
    defer zone.end();
    assert(framebuf_extent.width != 0 and framebuf_extent.height != 0);
    assert(self.in_frame);
    return Backend.acquireNextImage(self, framebuf_extent);
}

/// Will blocks until the next frame in flight's resources can be reclaimed.
pub fn beginFrame(self: *@This()) void {
    const zone = Zone.begin(.{
        .src = @src(),
        .color = gpu.global_options.blocking_zone_color,
    });
    defer zone.end();
    assert(!self.in_frame);
    self.in_frame = true;
    Backend.beginFrame(self);
    self.tracy_queries[self.frame] = 0;
}

/// Waits until the device is idle. Not recommended for common use, but may be useful for debugging
/// synchronization issues, or waiting until it's safe to exit in debug mode. In release mode you
/// should probably be using `std.process.cleanExit`.
pub fn waitIdle(self: *const @This()) void {
    Backend.waitIdle(self);
}
