//! See `ModTimer`.

/// A looping timer. Useful for shader effects that need time as an input, but don't want to run out
/// of precision when the game has been running for a while.
pub const ModTimer = extern struct {
    /// The period of the timer.
    ///
    /// Defaulting to 1000 allows periodic periodic effects to have up to three decimals of
    /// precision in their frequency when measured in seconds without causing a hitch when the timer
    /// resets.
    period: f32 = 1000,

    /// The current value of the timer in seconds.
    ///
    /// By initializing to a value near the point where the timer wraps, we're more likely to catch
    /// effects that don't correctly handle the wrap.
    seconds: f32 = 1000 - 5,

    /// Update the timer.
    pub fn update(self: *@This(), delta_s: f32) void {
        self.seconds = @rem(self.seconds + delta_s, self.period);
    }
};
