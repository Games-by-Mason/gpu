//! Mirrors of the hash functions provided by GBMS, see GBMS for documentation, origins, recommended
//! usage, etc:
//!
//! https://github.com/Games-by-Mason/gbms/
//!
//! Definitely not cryptographically secure. Probably not good in hash maps either. Tuned for
//! graphics.

const std = @import("std");

fn pcg(s: u32) u32 {
    const state: u32 = s *% 747796405 +% 2891336453;
    const word: u32 = ((state >> @intCast((state >> 28) +% 4)) ^ state) *% 277803737;
    return (word >> 22) ^ word;
}

// XXX: actually test these visually? with noise maybe?
// XXX: add tests just to make sure that they compile at least
// XXX: maybe also check soem random values for overflow
test pcg {
    // Check single value
    try std.testing.expectEqual(129708002, pcg(0));

    // Run it a few times to check for overflow
    var result: u32 = 0;
    for (0..1000) |_| {
        result = pcg(result);
    }
}

fn pcg2d(s: @Vector(2, u32)) @Vector(2, u32) {
    var r = s;
    r = r *% @as(@Vector(2, u32), @splat(1664525)) +% @as(@Vector(2, u32), @splat(1013904223));
    r[0] +%= r[1] *% 1664525;
    r[1] +%= r[0] *% 1664525;
    r = r ^ (r >> @splat(16));
    r[0] +%= r[1] *% 1664525;
    r[1] +%= r[0] *% 1664525;
    r = r ^ (r >> @splat(16));
    return r;
}

test pcg2d {
    // Check single value
    try std.testing.expectEqual(
        .{ 2313183303, 4026777116 },
        pcg2d(.{ 0, 1 }),
    );

    // Run it a few times to check for overflow
    var result: @Vector(2, u32) = @splat(0);
    for (0..1000) |_| {
        result = pcg2d(result);
    }
}

fn pcg3d(s: @Vector(3, u32)) @Vector(3, u32) {
    var r = s;
    r = r *% @as(@Vector(3, u32), @splat(1664525)) +% @as(@Vector(3, u32), @splat(1013904223));
    r[0] +%= r[1] *% r[2];
    r[1] +%= r[2] *% r[0];
    r[2] +%= r[0] *% r[1];
    r ^= r >> @splat(16);
    r[0] +%= r[1] *% r[2];
    r[1] +%= r[2] *% r[0];
    r[2] +%= r[0] *% r[1];
    return r;
}

test pcg3d {
    // Check single value
    try std.testing.expectEqual(
        .{ 3409911753, 1806500465, 2000842263 },
        pcg3d(.{ 0, 1, 2 }),
    );

    // Run it a few times to check for overflow
    var result: @Vector(3, u32) = @splat(0);
    for (0..1000) |_| {
        result = pcg3d(result);
    }
}

fn pcg4d(s: @Vector(4, u32)) @Vector(4, u32) {
    var r = s;
    r = r *% @as(@Vector(4, u32), @splat(1664525)) +% @as(@Vector(4, u32), @splat(1013904223));
    r[0] +%= r[1] *% r[3];
    r[1] +%= r[2] *% r[0];
    r[2] +%= r[0] *% r[1];
    r[3] +%= r[1] *% r[2];
    r ^= r >> @splat(16);
    r[0] +%= r[1] *% r[3];
    r[1] +%= r[2] *% r[0];
    r[2] +%= r[0] *% r[1];
    r[3] +%= r[1] *% r[2];
    return r;
}

test pcg4d {
    // Check single value
    try std.testing.expectEqual(
        .{ 933639991, 2252496459, 1324331964, 1990701187 },
        pcg4d(.{ 0, 1, 2, 3 }),
    );

    // Run it a few times to check for overflow
    var result: @Vector(4, u32) = @splat(0);
    for (0..1000) |_| {
        result = pcg4d(result);
    }
}

// XXX: test these
fn default1D(s: u32) u32 {
    return pcg(s);
}

test default1D {
    // Make sure it compiles
    _ = default1D(0);
}

fn default2D(s: @Vector(2, u32)) @Vector(2, u32) {
    return pcg2d(s);
}

test default2D {
    // Make sure it compiles
    _ = default2D(.{ 0, 1 });
}

fn default3D(s: [3]u32) [3]u32 {
    return pcg3d(s);
}

test default3D {
    // Make sure it compiles
    _ = default3D(.{ 0, 1, 2 });
}

fn default4D(s: @Vector(4, u32)) @Vector(4, u32) {
    return pcg4d(s);
}

test default4D {
    // Make sure it compiles
    _ = default4D(.{ 0, 1, 2, 3 });
}
