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
const Image = gpu.Image;

const Ctx = @This();

const Backend = gpu.global_options.Backend;

/// Backend specific state.
backend: Backend,
/// Device information, initialized at init time.
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
    pub const SurfaceFormat = enum {
        /// Requests a 4 channel 8 bit per channel unorm surface format with any channel order.
        ///
        /// These formats do no automatic conversion of color spaces, they're the right choice when
        /// your shaders write sRGB colors.
        unorm4x8,
        /// Requests a 4 channel 8 bit per channel sRGB surface format with any channel order.
        ///
        /// These formats convert from linear to sRGB on write. They're the right choice when your
        /// shaders write linear colors.
        srgb4x8,
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
        var ranks = std.EnumArray(Device.Kind, u8).initFill(1);
        ranks.set(.discrete, 3);
        ranks.set(.integrated, 2);
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
    /// The device type rankings. Higher ranked device types are prioritized, rank 0 devices are
    /// skipped.
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
    /// The surface format request. This request will resolve to a format supported by the current
    /// hardware, you can check the result at `device.surface_format`.
    surface_format: SurfaceFormat,
    /// The initial surface extent.
    surface_extent: Extent2D,
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

/// Options for `endFrame`.
pub const EndFrameOptions = struct {
    // At the end of the frame, if provided, an image is blitted to the swapchain.
    //
    // At first glance this may appear wasteful--why not write directly to the swapchain? In
    // practice, especially on PC, you don't have much control over the swapchain resolution, but
    // you very much care about your render resolution. This means that you really want to be
    // rendering to your own buffer, and then copying it to the swapchain at the end.
    //
    // Furthermore, on some backends (like DX12) you can't write to the swapchain from compute
    // shaders, which means you'd end up needing this extra copy anyway.
    //
    // In practice this is *very* cheap, and also allows us to defer acquiring the swapchain image/
    // waiting on the present semaphore as long as possible, so is likely a performance win in
    // practice due to increased pipelining.
    //
    // Additionally, this allows us to present a simpler API to the caller: they can just provide us
    // with an image, instead of needing to worry about synchronization around the swapchain image.
    // Assuming the proper layout transitions, the rest can be handled automatically by us.
    //
    // The main downside to this approach is slightly increased memory usage. This library is
    // designed for desktop and console, though, and so this very marginal increase is not expected
    // to be a problem.
    pub const Present = struct {
        /// The image to present. Must be in the "present blit" layout.
        handle: gpu.ImageHandle,
        /// The extent of the source image to present.
        src_extent: Extent2D,
        /// A hint for the extent of the surface to present to, on PC you typically pass the window
        /// size.
        ///
        /// Note that on some window protocols querying the window extent can be surprisingly
        /// expensive, if it's available as an event on change you're typically better off caching
        /// the value from the event.
        surface_extent: Extent2D,
        /// The range of the source image to present.
        range: gpu.ImageRange,
        /// The filter to use when blitting the source to the swapchain.
        filter: gpu.ImageFilter,
    };

    /// What to present if we're rendering this frame. May be set to null e.g. on setup frames.
    present: ?Present,
};

/// Ends the current frame.
///
/// Drivers may sometimes block here instead of on `acquireNextImage`.
pub fn endFrame(self: *@This(), options: EndFrameOptions) void {
    const zone = Zone.begin(.{ .src = @src() });
    defer zone.end();
    if (options.present) |present| {
        assert(present.src_extent.width != 0 and present.src_extent.height != 0);
        assert(present.surface_extent.width != 0 and present.surface_extent.height != 0);
        assert(self.in_frame);
    }
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

/// The result of `acquireNextImage`.
pub const AcquireNextImageResult = struct {
    image: gpu.Image(.color),
    extent: Extent2D,
};

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

/// Submits the given command buffers to the GPU in order.
pub fn submit(self: *@This(), cbs: []const gpu.CmdBuf) void {
    const zone = Zone.begin(.{ .src = @src() });
    defer zone.end();
    assert(self.in_frame);
    Backend.submit(self, cbs);
}

/// Submit descriptor set update commands. Fastest when sorted. Copy commands not currently
/// supported.
pub fn updateDescSets(self: *@This(), updates: []const DescSet.Update) void {
    if (updates.len == 0) return;
    Backend.descSetsUpdate(self, updates);
}

/// Waits until the device is idle. Not recommended for common use, but may be useful for debugging
/// synchronization issues, or waiting until it's safe to exit in debug mode. In release mode you
/// should probably be using `std.process.cleanExit`.
pub fn waitIdle(self: *const @This()) void {
    Backend.waitIdle(self);
}
