//! A looping timer for shader effects. Looping prevents

/// We loop at 1000 seconds, which allows periodic effects to have up to three decimals worth of
/// precision in their frequency without causing a hitch when the timer wraps.
pub const max = 1000;

/// By starting near the point where the timer loops, we're more likely to catch effects that don't
/// handle the wrap correctly.
seconds: f32 = max - 10,

/// Update the timer.
pub fn update(self: *@This(), delta_s: f32) void {
    self.seconds = @rem(self.seconds + delta_s, max);
}
