//! A frame pacer. Reduces latency, and smooths delta time.
//!
//! # Problem Statement
//!
//! Many games implement variable timestep by scaling velocities by delta time, where delta time is
//! calculated as the time that has elapsed between the start of the last frame and the start of the
//! current frame.
//!
//! This is simple, but unfortunately, it's also incorrect. Not imprecise--incorrect.
//!
//! ## What's wrong with the common approach?
//!
//! This strategy is not measuring the current frame's delta time, it's measuring the last frame's
//! delta time, and neither of these values are what you want to multiply your velocity by. For
//! smooth animation, the correct delta to multiply by is the time difference between the display
//! time of your last frame, and the upcoming display time of your current frame.
//!
//! This sounds pedantic, but it's not. Getting this wrong results in visual artifacts in the form
//! of judder. The faster your game is, the worse the judder gets, and implemented naively the
//! further you get from optimal input latency.
//!
//! Why input latency? Because if you're using a FIFO presentation mode and vsync, then once the
//! swapchain image queue is full, you'll block waiting for the next vblank. You're blocking *after*
//! having polled for input and rendered the frame, increasing the time between the player's input
//! ocurring and being displayed on screen.
//!
//! ## In a perfect world...
//!
//! In a perfect world, the underlying system would provide us with:
//!
//! - The cutoff to submit our frame and make the next vblank
//! - The time the next frame will be displayed on screen
//! - The time the last frame was displayed on screen
//! - The current depth of the swapchain queue
//!
//! It would be trivial to use this in conjunction with timer queries to calculate the perfect delta
//! time and correct amount to sleep to minimize input latency.
//!
//! Unfortunately, everyone who works on these systems is too busy trying to cash in on crytpo or
//! LLMs or whatever the latest fad is to implement these basic features that would make literally
//! every game run better, so we're going to have to do some estimating and hope for the best.
//!
//! # Why not use `VK_AMD_anti_lag`/`VK_NV_low_latency2`?
//!
//! In theory, these extensions should at least help with lowering latency on supported cards.
//!
//! In practice, I've found that both extensions only work when using their respective FPS limiters
//! to set set an FPS cap *lower* than that of the frame. It probably goes without saying that this
//! is unacceptable because it forces you to repeatedly miss vblank, leading ot massive judder.
//!
//! In light of this, I can't imagine why you would ever opt into either of these extensions. If you
//! have any idea what's going on here contact me.
//!
//! # Usage
//!
//! ## Basic Usage: Smoothed Delta Time
//!
//! Your game should create an instance of `FramePacer` immediately before the game loop starts, and
//! call `update` on it immediately after each presentation is submitted.
//!
//! Internally, the frame pacer will keep track of how often it is called, and store a smoothed
//! delta time value on `smoothed_delta_s`. This is the value you should pass into the rest of your
//! engine to e.g. multiply velocities by.
//!
//! This value is calculated by timing how often update is called, and returning an average with a
//! bias towards recent calls. This mitigates variability in update time due to the swapchain
//! queueing up frames, and the fact that you can't trust the refresh rate reported by the OS (it
//! may be unavailable, or flat out wrong in the case of VRR.)
//!
//! You should still pass in the refresh rate on init if you have it since this prewarms the
//! average, and you update it by assigning to `refresh_rate_hz` if you get an event from the OS
//! notifying you that it changed as it's required for some more advanced features. Don't poll the
//! OS for the refresh rate every frame, on some platforms this is surprisingly effective.
//!
//! # Advanced Usage: Smoothed Delta Time + Input Latency Reduction
//!
//! The basic usage will get you smoother delta times, but it won't help with input latency.
//!
//! Lower input latency comes with the risk of judder if you push it too far, so you may want to
//! allow users to control whether or not this part of the integration is enabled.
//!
//! First off, when you desire low input latency, you should set your swapchain image count to the
//! minimum value by calling `gx.setLowLatency(true)`, or enabling low latency mode on init.
//!
//! Next, you may have noticed that `update` takes an argument labeled `slop_ns`, and returns a
//! `u64`. Slop is the amount of time the CPU spent blocked on the GPU this frame (e.g. because the
//! swapchain queue was full.) Our goal is to move that time spent blocking to before we poll for
//! input. You can get this from `gx.slop_ns`.
//!
//! The value returned by `update` is the number of nanoseconds the frame pacer is recommending that
//! you wait before polling input. You can sleep during this time (e.g. with something like
//! `SDL_DelayPrecise`), or do other useful work that doesn't depend on user input.
//!
//! The difference is the most noticable at 60hz. The higher your refresh rate, the lower input
//! latency gets naturally since you're polling more often. You may want to use something like
//! [`GBMS`](https://github.com/Games-by-Mason/gbms/)'s latency test shader to observe the
//! difference.
//!
//! The default options are reasonable. You may notice that you can save an extra 16ms of latency
//! by setting the headroom near 0 if your game render's very fast, but you don't want to do this--
//! you're going to reintroduce judder if there are even minor timing variations as you'll be able
//! to see in something like the GBMS latency tester.

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
/// The smoothed delta time. Updated by `sleep`.
smoothed_delta_s: f32,
/// The frame timer.
timer: std.time.Timer,

const plot_smoothed_delta_s = "FP: Smoothed Delta S";
const plot_delta_s = "FP: Delta S";
const plot_slop_ms = "FP: Slop MS";
const plot_sleep_ms = "FP: Sleep MS";

const plot_names: []const [:0]const u8 = &.{
    plot_smoothed_delta_s,
    plot_delta_s,
    plot_slop_ms,
    plot_sleep_ms,
};

/// You may pass `0` in for the refresh rate if it is unknown. This will disable the latency
/// reduction and the smoothed delta time will take slightly longer to converge.
///
/// It's tempting to make this parameter nullable instead of using in band signaling. In practice,
/// many operating systems and platform layers already use 0 as "unknown".
pub fn init(refresh_rate_hz: f32) @This() {
    for (plot_names) |name| {
        tracy.plotConfig(.{
            .name = name,
            .format = .number,
            .mode = .line,
            .fill = true,
        });
    }

    return .{
        .refresh_rate_hz = refresh_rate_hz,
        .smoothed_delta_s = b: {
            if (refresh_rate_hz == 0) {
                // Prewarm to 60hz in absence of a known refresh rate
                break :b 1.0 / 60.0;
            } else {
                // Prewarm to the specified refresh rate. It may not be correct (e.g. in the case of
                // VRR) but it's a good starting point.
                break :b 1.0 / refresh_rate_hz;
            }
        },
        .timer = std.time.Timer.start() catch |err| @panic(@errorName(err)),
    };
}

/// Slop is the amount of time the CPU spent blocked on the GPU this frame, you can get this from
/// `gx.slop_ns`. Alternatigvely you may pass in `0` if this value is unknown for some reason or you
/// don't care about lowering input latency, this wil disable input latency reduction. The return
/// value is the suggested number of nanoseconds to delay before polling for user input.
pub fn update(self: *@This(), slop_ns: u64) u64 {
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

    // Calculate the recommended sleep time before input latency.
    const sleep_ns: u64 = b: {
        // Early out if we don't know the target refresh rate. Without it, we have no way to bound
        // our sleep amount in the event that our headroom is too low on a given platform and will
        // death spiral, so we don't request any sleep.
        if (refresh_period_ms == 0) {
            self.sleep_ms = 0;
            break :b 0;
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

        // Break with the recommended delay in nanoseconds
        break :b @intFromFloat(self.sleep_ms * std.time.ns_per_ms);
    };

    // Tracy plots
    tracy.plot(.{
        .name = plot_slop_ms,
        .value = .{ .f32 = slop_ms },
    });
    tracy.plot(.{
        .name = plot_delta_s,
        .value = .{ .f32 = delta_s },
    });
    tracy.plot(.{
        .name = plot_smoothed_delta_s,
        .value = .{ .f32 = self.smoothed_delta_s },
    });
    tracy.plot(.{
        .name = plot_sleep_ms,
        .value = .{ .f32 = self.sleep_ms },
    });

    return sleep_ns;
}
