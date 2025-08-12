//! A frame pacer. Reduces latency and smooths delta time.

// XXX: TODO:
// - document rational, explain what the ideal version of this would look like and why we can't do that
// - explain why we can't use amd/nvidia's implementaitons
// - note limitaitons around vrr/frame rate limiting
// - simplify getting the blocked value from gx, probably store the last frame's blocked value
// isntead of returning
// - test on all 4 test setups, test with heavier gpu load
// - probably turn these docs into a blog post (and ask for feedback!)

const std = @import("std");
const geom = @import("geom");
const tracy = @import("tracy");

const lerp = geom.tween.interp.lerp;

const Zone = tracy.Zone;

/// The target frame time, or `0` for no latency reduction. You may update this value if the monitor
/// changes. Note that querying this value from some platforms is slow, prefer only polling for a
/// new value in response to a display changed event if possible.
refresh_rate_hz: f32,
/// The amount of headroom to leave. Configurable.
headroom_ms: f32 = 0.5,
/// If a frame time overshoots the target by more than this much, scale back the sleep amount.
overshoot_ms: f32 = 1.0,
/// The amount to scale back the sleep on overshoot.
overshoot_scale: f32 = 0.9,
/// The current sleep amount. Updated by `sleep`.
sleep_ms: f32 = 0.0,
/// The ratio of the slop to convert to sleep each update.
sleep_rwa: f32 = 0.1,
/// The ratio of the current frame time to factor into the smoothed delta time every frame.
smoothed_rwa: f32 = 0.1,
/// The max smooth frame time.
max_smoothed_s: f32 = 1.0 / 30.0,
/// The smoothed delta time. Updated by `sleep`, pre-warmed to a resaonable value.
smoothed_delta_s: f32 = 1.0 / 60.0,
/// The frame timer.
timer: std.time.Timer,

pub fn init(refresh_rate_hz: f32) @This() {
    return .{
        .refresh_rate_hz = refresh_rate_hz,
        .timer = std.time.Timer.start() catch |err| @panic(@errorName(err)),
    };
}

pub fn sleep(self: *@This(), slop_ns: u64) void {
    // Lap the frame timer, conver to useful units.
    const delta_ns = self.timer.lap();

    // Unit conversions
    const slop_ms: f32 = @as(f32, @floatFromInt(slop_ns)) / std.time.ns_per_ms;
    const delta_s = @as(f32, @floatFromInt(delta_ns)) / std.time.ns_per_s;
    const delta_ms = @as(f32, @floatFromInt(delta_ns)) / std.time.ns_per_ms;
    const refresh_period_ms = if (self.refresh_rate_hz == 0) 0 else 1000.0 / self.refresh_rate_hz;

    // Update the smoothed delta time.
    self.smoothed_delta_s = lerp(
        self.smoothed_delta_s,
        @min(delta_s, self.max_smoothed_s),
        self.smoothed_rwa,
    );

    // Early out if we don't know the target refresh rate. Without it, we have no way to bound
    // our sleep amount in the event that our headroom is too low on a given platform and will
    // death spiral.
    if (refresh_period_ms == 0) {
        return;
    }

    // If our frame time overshot our max, scale back our sleep amount. Ideally this will never
    // happen. Under normal circumstnaces, scale towards leaving headroom slop only.
    if (delta_ms > refresh_period_ms + self.overshoot_ms) {
        self.sleep_ms *= self.overshoot_scale;
    } else {
        const diff = slop_ms - self.headroom_ms;
        self.sleep_ms += diff * self.sleep_rwa;
    }

    // Clamp to the headroom allowed range.
    self.sleep_ms = std.math.clamp(
        self.sleep_ms,
        0.0,
        @max(refresh_period_ms - self.headroom_ms, 0.0),
    );

    // Sleep. If we're sleeping for a while we'll actually put the thread to sleep, but then
    // we wake up and busy wait before the timer is up to decrease the risk of sleeping too
    // long. This might be overly cautious.
    if (self.sleep_ms > 0) {
        const zone = Zone.begin(.{ .name = "pacer sleep", .src = @src() });
        defer zone.end();
        const sleep_ns: u64 = @intFromFloat(self.sleep_ms * std.time.ns_per_ms);
        var sleep_timer = std.time.Timer.start() catch |err| @panic(@errorName(err));
        if (self.sleep_ms > 5) std.Thread.sleep(sleep_ns * 2 / 3);
        while (sleep_timer.read() < sleep_ns) {}
    }
}
