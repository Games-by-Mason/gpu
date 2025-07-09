//! Gaussian blurs are useful for post processing, but making them fast is a bit tricky.
//!
//! The first key to making Gaussian blurs fast is realizing they're separable, e.g. you can blur on
//! the X and Y axis separately. This makes a O(N^2) operation into an O(N) operation which is a big
//! improvement. Similarly, if you downscale before blurring by 2x, you reduce the number of samples
//! you need to take by 4x.
//!
//! That's the low hanging fruit, but it's possible to go faster...
//!
//! # Large Gaussian Blurs
//!
//! You can approximate a Gaussian blur with a repeated box blur (2 or 3 passes per dimension is
//! typically enough), and a box blur can be computed as a moving average in a compute shader. This
//! lets you do extremely large blurs very cheap since the blur radius has little effect on the
//! final performance.
//!
//! This can be implemented entirely in a compute shader, no help from this extension is necessary.
//!
//! # Linear Filtering
//!
//! For smaller blurs, the up front cost of the moving average blur is unnecessarily expensive. For
//! these you can instead take advantage of hardware filtering. By sampling in between pixels with
//! linear filtering enabled, you can sample up to 4 pixels for essentially the price of one.
//!
//! If you carefully choose your weights, this lets us halve the number of samples necessary for our
//! separable Gaussian kernels. Calculating those weights is a bit annoying, though, so this
//! extension provides a helper to do it for you.
//!
//! Note that it's technically possible to go even faster by sampling in between 4 pixels instead of
//! 2, but it's a significantly harder to create a resolution independent blur this way. If that
//! isn't an issue for your use case and you don't mind a lower quality blur, you'll want to look up
//! "Kawase Blur".
//!
//! # Additional Reading
//! * Kawase Blur: https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-20-66/siggraph2015_2D00_mmg_2D00_marius_2D00_notes.pdf
//! * Blur performance comparisons: https://www.intel.com/content/www/us/en/developer/articles/technical/an-investigation-of-fast-real-time-gpu-based-image-blur-algorithms.html
//! * Small Sigma Correction (not currently implemented here): https://bartwronski.com/2021/10/31/practical-gaussian-filter-binomial-filter-and-small-sigma-gaussians/

const std = @import("std");

/// Options for `linear`.
const LinearOptions = struct {
    weights_buf: []f32,
    offsets_buf: []f32,
    threshold: f32 = 0.5 / 255.0,
    sigma: f32,
};

/// The result of `linear`.
pub const Linear = struct {
    /// The weight for each sample.
    weights: []f32,
    /// The offset from the center of the kernel for each sample. You can mirror this to create a
    /// one dimensional blur.
    ///
    /// Note that in a language like glsl, the lower left pixel is at vec2(0.5) not vec2(0), so
    /// you'll likely need to offset this by vec2(0.5) in yuour shader.
    offsets: []f32,
    /// Whether or not the buffer lengths lead to truncating the blur early.
    truncated: bool,
};

/// Returns weights and offsets to calculate a Gaussian blur using linear filtering.
pub fn linear(options: LinearOptions) Linear {
    const exp_scale = expScale(options.sigma);
    const max_len = @min(options.weights_buf.len, options.offsets_buf.len);
    var result: Linear = .{
        .weights = options.weights_buf[0..0],
        .offsets = options.offsets_buf[0..0],
        .truncated = true,
    };

    if (max_len > 0) {
        const w0 = weight(0, exp_scale);
        if (w0 > options.threshold) {
            var sum: f32 = w0;
            result.weights.len += 1;
            result.offsets.len += 1;
            result.weights[0] = w0;
            result.offsets[0] = 0;

            for (1..max_len) |x| {
                const w1 = weight(@floatFromInt(2 * x), exp_scale);
                const w2 = weight(@floatFromInt(2 * x + 1), exp_scale);
                const w12 = w1 + w2;

                const new_sum = @mulAdd(f32, 2, w12, sum);
                if (w12 / new_sum < options.threshold) {
                    result.truncated = false;
                    break;
                }
                sum = new_sum;

                result.weights.len += 1;
                result.offsets.len += 1;
                result.weights[x] = w12;
                result.offsets[x] = @as(f32, @floatFromInt(x * 2 - 1)) + w2 / (w12);
            }

            for (result.weights) |*w| w.* /= sum;
        }
    }

    return result;
}

/// Returns the exponent scale for a given sigma.
pub fn expScale(sigma: f32) f32 {
    return 1.0 / (sigma * sigma);
}

/// Returns the unnormalized weight for a given distance and exponent scale. Does not currently
/// support [small sigma correction](https://bartwronski.com/2021/10/31/practical-gaussian-filter-binomial-filter-and-small-sigma-gaussians/).
pub fn weight(x: f32, exp_scale: f32) f32 {
    return std.math.exp(-x * x * exp_scale);
}

test linear {
    var buf: [28]f32 = undefined;

    {
        const l0 = linear(.{
            .weights_buf = buf[0..14],
            .offsets_buf = buf[14..28],
            .sigma = 8,
        });
        try std.testing.expectEqualSlices(f32, &.{
            0.08208174,
            0.14842252,
            0.11946461,
            0.08494033,
            0.053348407,
            0.029597757,
            0.014505151,
            0.006279241,
            0.0024010886,
        }, l0.weights);
        try std.testing.expectEqualSlices(f32, &.{
            0,
            1.4804786,
            3.4649017,
            5.449393,
            7.4339814,
            9.418697,
            11.403566,
            13.388618,
            15.373876,
        }, l0.offsets);
        try std.testing.expect(!l0.truncated);
    }

    {
        const l0 = linear(.{
            .weights_buf = buf[0..14],
            .offsets_buf = buf[14..28],
            .sigma = 16,
        });
        try std.testing.expect(l0.truncated);
    }
}
