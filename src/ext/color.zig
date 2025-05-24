//! Convenience functions for color conversions.

const std = @import("std");
const assert = std.debug.assert;
const math = std.math;

pub fn linearToSrgb(linear: anytype) @TypeOf(linear) {
    const Color = @TypeOf(linear);
    switch (@typeInfo(Color)) {
        .float, .comptime_float => {
            // The color component transfer function from the SRGB specification:
            // https://www.w3.org/Graphics/Color/srgb
            if (linear <= 0.031308) {
                return 12.92 * linear;
            } else {
                return 1.055 * math.pow(f32, linear, 1.0 / 2.4) - 0.055;
            }
        },
        inline .array, .@"struct" => |info| {
            if (@typeInfo(Color) == .@"array" or info.is_tuple) {
                var srgb: Color = undefined;
                const len = if (@typeInfo(Color) == .array) info.len else info.fields.len;
                if (len > 0) srgb[0] = linearToSrgb(linear[0]);
                if (len > 1) srgb[1] = linearToSrgb(linear[1]);
                if (len > 2) srgb[2] = linearToSrgb(linear[2]);
                if (len > 3) srgb[3] = linear[3];
                comptime assert(len <= 4);
                comptime assert(len != 2); // Unclear if the second component is alpha or not!
                return srgb;
            } else {
                inline for (info.fields) |field| {
                    comptime assert(
                        std.mem.eql(u8, field.name, "r") or
                        std.mem.eql(u8, field.name, "g") or
                        std.mem.eql(u8, field.name, "b") or
                        std.mem.eql(u8, field.name, "a"));
                }
                var srgb: Color = undefined;
                if (@hasField(Color, "r")) srgb.r = linearToSrgb(linear.r);
                if (@hasField(Color, "g")) srgb.g = linearToSrgb(linear.g);
                if (@hasField(Color, "b")) srgb.b = linearToSrgb(linear.b);
                if (@hasField(Color, "a")) srgb.a = linear.a;
                return srgb;
            }
        },
        else => comptime unreachable,
    }
}

pub fn srgbToLinear(srgb: anytype) @TypeOf(srgb) {
    const Color = @TypeOf(srgb);
    switch (@typeInfo(Color)) {
        .float, .comptime_float => {
            // The inverse of the color component transfer function from the SRGB specification:
            // https://www.w3.org/Graphics/Color/srgb
            if (srgb <= 0.04045) {
                return srgb / 12.92;
            } else {
                return math.pow(f32, (srgb + 0.055) / (1.055), 2.4);
            }
        },
        inline .array, .@"struct" => |info| {
            if (@typeInfo(Color) == .@"array" or info.is_tuple) {
                var linear: Color = undefined;
                const len = if (@typeInfo(Color) == .array) info.len else info.fields.len;
                if (len > 0) linear[0] = srgbToLinear(srgb[0]);
                if (len > 1) linear[1] = srgbToLinear(srgb[1]);
                if (len > 2) linear[2] = srgbToLinear(srgb[2]);
                if (len > 3) linear[3] = srgb[3];
                comptime assert(len <= 4);
                comptime assert(len != 2); // Unclear if the second component is alpha or not!
                return linear;
            } else {
                inline for (info.fields) |field| {
                    comptime assert(
                        std.mem.eql(u8, field.name, "r") or
                        std.mem.eql(u8, field.name, "g") or
                        std.mem.eql(u8, field.name, "b") or
                        std.mem.eql(u8, field.name, "a"));
                }
                var linear: Color = undefined;
                if (@hasField(Color, "r")) linear.r = srgbToLinear(srgb.r);
                if (@hasField(Color, "g")) linear.g = srgbToLinear(srgb.g);
                if (@hasField(Color, "b")) linear.b = srgbToLinear(srgb.b);
                if (@hasField(Color, "a")) linear.a = srgb.a;
                return linear;
            }
        },
        else => comptime unreachable,
    }
}

test "srgb" {
    // Comptime floats
    try std.testing.expectApproxEqRel(0.5, @as(f32, srgbToLinear(linearToSrgb(0.5))), 0.00001);

    // Floats
    try std.testing.expectApproxEqRel(0.5, srgbToLinear(linearToSrgb(@as(f32, 0.5))), 0.00001);

    // Structs with alpha
    {
        const Color = struct {
            r: f32,
            g: f32,
            b: f32,
            a: f32,
        };
        const result = srgbToLinear(linearToSrgb(Color { .r = 0.2, .g = 0.3, .b = 0.4, .a = 0.5 }));
        try std.testing.expectApproxEqRel(0.2, result.r, 0.00001);
        try std.testing.expectApproxEqRel(0.3, result.g, 0.00001);
        try std.testing.expectApproxEqRel(0.4, result.b, 0.00001);
        try std.testing.expectEqual(0.5, result.a);
    }

    // Structs without alpha
    {
        const Color = struct {
            r: f32,
            g: f32,
            b: f32,
        };
        const result = srgbToLinear(linearToSrgb(Color { .r = 0.2, .g = 0.3, .b = 0.4 }));
        try std.testing.expectApproxEqRel(0.2, result.r, 0.00001);
        try std.testing.expectApproxEqRel(0.3, result.g, 0.00001);
        try std.testing.expectApproxEqRel(0.4, result.b, 0.00001);
    }

    // Tuples with alpha
    {
        const Color = struct {
            f32,
            f32,
            f32,
            f32,
        };
        const result = srgbToLinear(linearToSrgb(Color { 0.2, 0.3, 0.4, 0.5 }));
        try std.testing.expectApproxEqRel(0.2, result[0], 0.00001);
        try std.testing.expectApproxEqRel(0.3, result[1], 0.00001);
        try std.testing.expectApproxEqRel(0.4, result[2], 0.00001);
        try std.testing.expectEqual(0.5, result[3]);
    }

    // Tuples without alpha
    {
        const Color = struct {
            f32,
            f32,
            f32,
        };
        const result = srgbToLinear(linearToSrgb(Color { 0.2, 0.3, 0.4 }));
        try std.testing.expectApproxEqRel(0.2, result[0], 0.00001);
        try std.testing.expectApproxEqRel(0.3, result[1], 0.00001);
        try std.testing.expectApproxEqRel(0.4, result[2], 0.00001);
    }

    // Arrays with alpha
    {
        const result = srgbToLinear(linearToSrgb([4]f32 { 0.2, 0.3, 0.4, 0.5 }));
        try std.testing.expectApproxEqRel(0.2, result[0], 0.00001);
        try std.testing.expectApproxEqRel(0.3, result[1], 0.00001);
        try std.testing.expectApproxEqRel(0.4, result[2], 0.00001);
        try std.testing.expectEqual(0.5, result[3]);
    }

    // Arrays without alpha
    {
        const result = srgbToLinear(linearToSrgb([3]f32 { 0.2, 0.3, 0.4 }));
        try std.testing.expectApproxEqRel(0.2, result[0], 0.00001);
        try std.testing.expectApproxEqRel(0.3, result[1], 0.00001);
        try std.testing.expectApproxEqRel(0.4, result[2], 0.00001);
    }
}
