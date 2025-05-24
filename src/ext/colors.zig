//! Convenience functions for color conversions. Methods operate on structs containing r/g/b/a
//! fields, tuples, arrays, or vectors.

const std = @import("std");
const assert = std.debug.assert;
const math = std.math;

/// The number of decimals that survive a conversion from f32 to 8 bit unorm. Provided for
/// documentation purposes, empirical proof in test.
pub const f32_to_u8_unorm_decimals = 3;

test f32_to_u8_unorm_decimals {
    for (0..255) |u| {
        const f = unormToFloat(f32, @as(u8, @intCast(u)));
        const truncated = @floor(@as(f128, f) * 1000.0) / 1000.0;
        const unorm = floatToUnorm(u8, truncated);
        try std.testing.expectEqual(u, unorm);
    }
}

/// Converts a linear float color to sRGB.
pub fn linearToSrgb(T: type, linear: T) T {
    var result = anyColor(linear);
    if (@TypeOf(result.r) != void) result.r = linearToSrgbInner(result.r);
    if (@TypeOf(result.g) != void) result.g = linearToSrgbInner(result.g);
    if (@TypeOf(result.b) != void) result.b = linearToSrgbInner(result.b);
    return result.to(T);
}

fn linearToSrgbInner(linear: anytype) @TypeOf(linear) {
    // The color component transfer function from the SRGB specification:
    // https://www.w3.org/Graphics/Color/srgb
    if (linear <= 0.031308) {
        return 12.92 * linear;
    } else {
        @setEvalBranchQuota(2000);
        return 1.055 * math.pow(@TypeOf(linear), linear, 1.0 / 2.4) - 0.055;
    }
}

/// Converts a sRGB float color to linear sRGB.
pub fn srgbToLinear(T: type, srgb: T) T {
    var result = anyColor(srgb);
    if (@TypeOf(result.r) != void) result.r = srgbToLinearInner(result.r);
    if (@TypeOf(result.g) != void) result.g = srgbToLinearInner(result.g);
    if (@TypeOf(result.b) != void) result.b = srgbToLinearInner(result.b);
    return result.to(T);
}

fn srgbToLinearInner(srgb: anytype) @TypeOf(srgb) {
    // The inverse of the color component transfer function from the SRGB specification:
    // https://www.w3.org/Graphics/Color/srgb
    if (srgb <= 0.04045) {
        return srgb / 12.92;
    } else {
        @setEvalBranchQuota(2000);
        return math.pow(@TypeOf(srgb), (srgb + 0.055) / 1.055, 2.4);
    }
}

test "srgb" {
    // With alpha
    {
        const result = srgbToLinear([4]f32, linearToSrgb([4]f32, .{ 0.2, 0.3, 0.4, 0.5 }));
        try std.testing.expectApproxEqRel(0.2, result[0], 0.00001);
        try std.testing.expectApproxEqRel(0.3, result[1], 0.00001);
        try std.testing.expectApproxEqRel(0.4, result[2], 0.00001);
        try std.testing.expectEqual(0.5, result[3]);
    }

    // Without alpha
    {
        const result = srgbToLinear([3]f32, linearToSrgb([3]f32, .{ 0.2, 0.3, 0.4 }));
        try std.testing.expectApproxEqRel(0.2, result[0], 0.00001);
        try std.testing.expectApproxEqRel(0.3, result[1], 0.00001);
        try std.testing.expectApproxEqRel(0.4, result[2], 0.00001);
    }
}

/// Converts a float color to a unorm color.
pub fn floatToUnorm(Result: type, input: anytype) Result {
    const any_input = anyColor(input);
    var result: AnyColor(Result) = undefined;
    if (@TypeOf(result.r) != void) result.r = floatToUnormInner(@TypeOf(result.r), any_input.r);
    if (@TypeOf(result.g) != void) result.g = floatToUnormInner(@TypeOf(result.g), any_input.g);
    if (@TypeOf(result.b) != void) result.b = floatToUnormInner(@TypeOf(result.b), any_input.b);
    if (@TypeOf(result.a) != void) result.a = floatToUnormInner(@TypeOf(result.a), any_input.a);
    return result.to(Result);
}

fn floatToUnormInner(Result: type, f: anytype) Result {
    return @intFromFloat(f * math.maxInt(Result) + 0.5);
}

test floatToUnorm {
    try std.testing.expectEqual(255, floatToUnorm(u8, 1.0));
    try std.testing.expectEqual(128, floatToUnorm(u8, 0.5));
    try std.testing.expectEqual(0, floatToUnorm(u8, 0.0));
}

/// Converts a unorm color to a float color.
pub fn unormToFloat(Result: type, input: anytype) Result {
    const any_input = anyColor(input);
    var result: AnyColor(Result) = undefined;
    if (@TypeOf(result.r) != void) result.r = unormToFloatInner(@TypeOf(result.r), any_input.r);
    if (@TypeOf(result.g) != void) result.g = unormToFloatInner(@TypeOf(result.g), any_input.g);
    if (@TypeOf(result.b) != void) result.b = unormToFloatInner(@TypeOf(result.b), any_input.b);
    if (@TypeOf(result.a) != void) result.a = unormToFloatInner(@TypeOf(result.a), any_input.a);
    return result.to(Result);
}

fn unormToFloatInner(Result: type, u: anytype) Result {
    if (Result == f32) {
        // It's slightly faster to multiply by the reciprocal, but it's inexact. Scaling the
        // numerator and denominator by 3 provides exact results for all f32s in this range as is
        // verified by tests.
        //
        // I wasn't able to quickly find scales that achieve this for other less common float types,
        // it may not be possible. This is likely fine, if we ever need to go faster here's a
        // good reference:
        //
        // https://fgiesen.wordpress.com/2024/12/24/unorm-and-snorm-to-float-hardware-edition/
        const max: Result = @floatFromInt(math.maxInt(@TypeOf(u)));
        const r: Result = 1.0 / (3.0 * max);
        return @as(Result, @floatFromInt(u)) * 3.0 * r;
    } else {
        return @as(Result, @floatFromInt(u)) / math.maxInt(@TypeOf(u));
    }
}

test unormToFloat {
    // Make sure all float types get exact results
    for (0..255) |u| {
        try std.testing.expectEqual(
            @as(f16, @floatFromInt(u)) / 255.0,
            unormToFloat(f16, @as(u8, @intCast(u))),
        );
        try std.testing.expectEqual(
            @as(f32, @floatFromInt(u)) / 255.0,
            unormToFloat(f32, @as(u8, @intCast(u))),
        );
        try std.testing.expectEqual(
            @as(f64, @floatFromInt(u)) / 255.0,
            unormToFloat(f64, @as(u8, @intCast(u))),
        );
        try std.testing.expectEqual(
            @as(f80, @floatFromInt(u)) / 255.0,
            unormToFloat(f80, @as(u8, @intCast(u))),
        );
        try std.testing.expectEqual(
            @as(f128, @floatFromInt(u)) / 255.0,
            unormToFloat(f128, @as(u8, @intCast(u))),
        );
    }
}

/// Helper function for converting a sRGB float color to a linear unorm color.
pub fn srgbToLinearUnorm(Output: type, Input: type, input: Input) Output {
    return floatToUnorm(Output, srgbToLinear(Input, input));
}

test srgbToLinearUnorm {
    try std.testing.expectEqual(
        [4]u8{ 1, 51, 141, 255 },
        srgbToLinearUnorm([4]u8, [4]f32, .{ 0.063, 0.486, 0.769, 1.0 }),
    );
}

/// A generic color type for use in writing generic conversions. Not recommended for general
/// application use, provide your own color type for this.
pub fn AnyColor(Input: type) type {
    comptime var R = void;
    comptime var G = void;
    comptime var B = void;
    comptime var A = void;

    switch (@typeInfo(Input)) {
        .float, .comptime_float, .int, .comptime_int => {
            R = Input;
        },
        inline .array, .vector => |info| {
            if (info.len > 0) R = info.child;
            if (info.len > 1) G = info.child;
            if (info.len > 2) B = info.child;
            if (info.len > 3) A = info.child;
            comptime assert(info.len != 2); // Unclear if the second component is alpha or not!
            comptime assert(info.len <= 4);
        },
        .@"struct" => |info| {
            if (info.is_tuple) {
                if (info.fields.len > 0) R = info.fields[0].type;
                if (info.fields.len > 1) G = info.fields[1].type;
                if (info.fields.len > 2) B = info.fields[2].type;
                if (info.fields.len > 3) A = info.fields[3].type;
                comptime assert(info.fields.len != 2); // Unclear if the second component is alpha or not!
                comptime assert(info.fields.len <= 4);
            } else {
                inline for (info.fields) |field| {
                    comptime assert(std.mem.eql(u8, field.name, "r") or
                        std.mem.eql(u8, field.name, "g") or
                        std.mem.eql(u8, field.name, "b") or
                        std.mem.eql(u8, field.name, "a"));
                }
                if (@hasField(Input, "r")) R = @FieldType(Input, "r");
                if (@hasField(Input, "g")) G = @FieldType(Input, "g");
                if (@hasField(Input, "b")) B = @FieldType(Input, "b");
                if (@hasField(Input, "a")) A = @FieldType(Input, "a");
            }
        },
        else => comptime unreachable,
    }

    return struct {
        r: R,
        g: G,
        b: B,
        a: A,

        pub fn to(self: @This(), T: type) T {
            // Verify the output is a valid color type
            _ = AnyColor(T);

            // Count how many channels we have
            comptime var channels = 0;
            inline for (std.meta.fields(@This())) |field| {
                if (field.type != void) {
                    channels += 1;
                }
            }

            // Make the conversion
            var result: T = undefined;
            switch (@typeInfo(T)) {
                .float, .comptime_float, .int, .comptime_int => {
                    comptime assert(channels == 1);
                    result = self.r;
                },
                inline .array, .vector => |info| {
                    comptime assert(channels == info.len);
                    if (info.len > 0) result[0] = self.r;
                    if (info.len > 1) result[1] = self.g;
                    if (info.len > 2) result[2] = self.b;
                    if (info.len > 3) result[3] = self.a;
                },
                .@"struct" => |info| {
                    if (info.is_tuple) {
                        comptime assert(channels == info.fields.len);
                        if (info.fields.len > 0) result[0] = self.r;
                        if (info.fields.len > 1) result[1] = self.g;
                        if (info.fields.len > 2) result[2] = self.b;
                        if (info.fields.len > 3) result[3] = self.a;
                    } else {
                        if (@hasField(T, "r")) result.r = self.r;
                        if (@hasField(T, "g")) result.g = self.g;
                        if (@hasField(T, "b")) result.b = self.b;
                        if (@hasField(T, "a")) result.a = self.a;
                    }
                },
                else => comptime unreachable,
            }
            return result;
        }
    };
}

/// Creates an instance of `AnyColor` from a user type.
pub fn anyColor(input: anytype) AnyColor(@TypeOf(input)) {
    const Result = AnyColor(@TypeOf(input));
    var result: Result = undefined;

    switch (@typeInfo(@TypeOf(input))) {
        .float, .comptime_float, .int, .comptime_int => {
            result.r = input;
        },
        inline .array, .vector => |info| {
            if (info.len > 0) result.r = input[0];
            if (info.len > 1) result.g = input[1];
            if (info.len > 2) result.b = input[2];
            if (info.len > 3) result.a = input[3];
        },
        .@"struct" => |info| {
            if (info.is_tuple) {
                if (info.fields.len > 0) result.r = input[0];
                if (info.fields.len > 1) result.g = input[1];
                if (info.fields.len > 2) result.b = input[2];
                if (info.fields.len > 3) result.a = input[3];
            } else {
                if (@hasField(@TypeOf(input), "r")) result.r = input.r;
                if (@hasField(@TypeOf(input), "g")) result.g = input.g;
                if (@hasField(@TypeOf(input), "b")) result.b = input.b;
                if (@hasField(@TypeOf(input), "a")) result.a = input.a;
            }
        },
        else => comptime unreachable,
    }

    return result;
}

test anyColor {
    // Comptime float
    {
        const input = 1.5;
        const any = anyColor(input);
        try std.testing.expectEqual(comptime_float, @TypeOf(any.r));
        try std.testing.expectEqual(input, any.r);
        try std.testing.expectEqual({}, any.g);
        try std.testing.expectEqual({}, any.b);
        try std.testing.expectEqual({}, any.a);
        try std.testing.expectEqual(input, any.to(comptime_float));
    }

    // Float
    {
        const input: f32 = 1.5;
        const any = anyColor(input);
        try std.testing.expectEqual(f32, @TypeOf(any.r));
        try std.testing.expectEqual(input, any.r);
        try std.testing.expectEqual({}, any.g);
        try std.testing.expectEqual({}, any.b);
        try std.testing.expectEqual({}, any.a);
        try std.testing.expectEqual(input, any.to(f32));
    }

    // Comptime int
    {
        const input = 10;
        const any = anyColor(input);
        try std.testing.expectEqual(comptime_int, @TypeOf(any.r));
        try std.testing.expectEqual(input, any.r);
        try std.testing.expectEqual({}, any.g);
        try std.testing.expectEqual({}, any.b);
        try std.testing.expectEqual({}, any.a);
        try std.testing.expectEqual(input, any.to(comptime_int));
    }

    // Int
    {
        const input: u8 = 10;
        const any = anyColor(input);
        try std.testing.expectEqual(u8, @TypeOf(any.r));
        try std.testing.expectEqual(input, any.r);
        try std.testing.expectEqual({}, any.g);
        try std.testing.expectEqual({}, any.b);
        try std.testing.expectEqual({}, any.a);
        try std.testing.expectEqual(input, any.to(u8));
    }

    // Tuple 0
    {
        const Tuple = struct {};
        const input: Tuple = .{};
        const any = anyColor(input);
        try std.testing.expectEqual({}, any.r);
        try std.testing.expectEqual({}, any.g);
        try std.testing.expectEqual({}, any.b);
        try std.testing.expectEqual({}, any.a);
        try std.testing.expectEqual(Tuple{}, any.to(Tuple));
    }

    // Tuple 1
    {
        const input = .{@as(f32, 1.0)};
        const any = anyColor(input);
        try std.testing.expectEqual(f32, @TypeOf(any.r));
        try std.testing.expectEqual(input[0], any.r);
        try std.testing.expectEqual({}, any.g);
        try std.testing.expectEqual({}, any.b);
        try std.testing.expectEqual({}, any.a);
        try std.testing.expectEqual(.{@as(f32, 1.0)}, any.to(struct { f32 }));
    }

    // Tuple 3
    {
        const input = .{ 1.0, 2.0, 3.0 };
        const any = anyColor(input);
        try std.testing.expectEqual(comptime_float, @TypeOf(any.r));
        try std.testing.expectEqual(1.0, any.r);
        try std.testing.expectEqual(comptime_float, @TypeOf(any.g));
        try std.testing.expectEqual(2.0, any.g);
        try std.testing.expectEqual(comptime_float, @TypeOf(any.b));
        try std.testing.expectEqual(3.0, any.b);
        try std.testing.expectEqual({}, any.a);
        try std.testing.expectEqual(input, any.to(@TypeOf(input)));
    }

    // Tuple 4
    {
        const input = .{ 1.0, 2.0, 3.0, 4.0 };
        const any = anyColor(input);
        try std.testing.expectEqual(comptime_float, @TypeOf(any.r));
        try std.testing.expectEqual(1.0, any.r);
        try std.testing.expectEqual(comptime_float, @TypeOf(any.g));
        try std.testing.expectEqual(2.0, any.g);
        try std.testing.expectEqual(comptime_float, @TypeOf(any.b));
        try std.testing.expectEqual(3.0, any.b);
        try std.testing.expectEqual(comptime_float, @TypeOf(any.a));
        try std.testing.expectEqual(4.0, any.a);
        try std.testing.expectEqual(input, any.to(@TypeOf(input)));
    }

    // Array 0
    {
        const input: [0]f32 = .{};
        const any = anyColor(input);
        try std.testing.expectEqual({}, any.r);
        try std.testing.expectEqual({}, any.g);
        try std.testing.expectEqual({}, any.b);
        try std.testing.expectEqual({}, any.a);
        try std.testing.expectEqual(input, any.to(@TypeOf(input)));
    }

    // Array 1
    {
        const input: [1]f32 = .{1.0};
        const any = anyColor(input);
        try std.testing.expectEqual(f32, @TypeOf(any.r));
        try std.testing.expectEqual(1.0, any.r);
        try std.testing.expectEqual({}, any.g);
        try std.testing.expectEqual({}, any.b);
        try std.testing.expectEqual({}, any.a);
        try std.testing.expectEqual(input, any.to(@TypeOf(input)));
    }

    // Array 3
    {
        const input: [3]f32 = .{ 1.0, 2.0, 3.0 };
        const any = anyColor(input);
        try std.testing.expectEqual(f32, @TypeOf(any.r));
        try std.testing.expectEqual(1.0, any.r);
        try std.testing.expectEqual(f32, @TypeOf(any.g));
        try std.testing.expectEqual(2.0, any.g);
        try std.testing.expectEqual(f32, @TypeOf(any.b));
        try std.testing.expectEqual(3.0, any.b);
        try std.testing.expectEqual({}, any.a);
        try std.testing.expectEqual(input, any.to(@TypeOf(input)));
    }

    // Array 4
    {
        const input: [4]f32 = .{ 1.0, 2.0, 3.0, 4.0 };
        const any = anyColor(input);
        try std.testing.expectEqual(f32, @TypeOf(any.r));
        try std.testing.expectEqual(1.0, any.r);
        try std.testing.expectEqual(f32, @TypeOf(any.g));
        try std.testing.expectEqual(2.0, any.g);
        try std.testing.expectEqual(f32, @TypeOf(any.b));
        try std.testing.expectEqual(3.0, any.b);
        try std.testing.expectEqual(f32, @TypeOf(any.a));
        try std.testing.expectEqual(4.0, any.a);
        try std.testing.expectEqual(input, any.to(@TypeOf(input)));
    }

    // Vector 0
    {
        const input: @Vector(0, f32) = .{};
        const any = anyColor(input);
        try std.testing.expectEqual({}, any.r);
        try std.testing.expectEqual({}, any.g);
        try std.testing.expectEqual({}, any.b);
        try std.testing.expectEqual({}, any.a);
        try std.testing.expectEqual(input, any.to(@TypeOf(input)));
    }

    // Vector 1
    {
        const input: @Vector(1, f32) = .{1.0};
        const any = anyColor(input);
        try std.testing.expectEqual(f32, @TypeOf(any.r));
        try std.testing.expectEqual(1.0, any.r);
        try std.testing.expectEqual({}, any.g);
        try std.testing.expectEqual({}, any.b);
        try std.testing.expectEqual({}, any.a);
        try std.testing.expectEqual(input, any.to(@TypeOf(input)));
    }

    // Vector 3
    {
        const input: @Vector(3, f32) = .{ 1.0, 2.0, 3.0 };
        const any = anyColor(input);
        try std.testing.expectEqual(f32, @TypeOf(any.r));
        try std.testing.expectEqual(1.0, any.r);
        try std.testing.expectEqual(f32, @TypeOf(any.g));
        try std.testing.expectEqual(2.0, any.g);
        try std.testing.expectEqual(f32, @TypeOf(any.b));
        try std.testing.expectEqual(3.0, any.b);
        try std.testing.expectEqual({}, any.a);
        try std.testing.expectEqual(input, any.to(@TypeOf(input)));
    }

    // Vector 4
    {
        const input: @Vector(4, f32) = .{ 1.0, 2.0, 3.0, 4.0 };
        const any = anyColor(input);
        try std.testing.expectEqual(f32, @TypeOf(any.r));
        try std.testing.expectEqual(1.0, any.r);
        try std.testing.expectEqual(f32, @TypeOf(any.g));
        try std.testing.expectEqual(2.0, any.g);
        try std.testing.expectEqual(f32, @TypeOf(any.b));
        try std.testing.expectEqual(3.0, any.b);
        try std.testing.expectEqual(f32, @TypeOf(any.a));
        try std.testing.expectEqual(4.0, any.a);
        try std.testing.expectEqual(input, any.to(@TypeOf(input)));
    }

    // Struct all
    {
        const input = .{ .r = 1.0, .g = 2.0, .b = 3.0, .a = 4.0 };
        const any = anyColor(input);
        try std.testing.expectEqual(comptime_float, @TypeOf(any.r));
        try std.testing.expectEqual(1.0, any.r);
        try std.testing.expectEqual(comptime_float, @TypeOf(any.g));
        try std.testing.expectEqual(2.0, any.g);
        try std.testing.expectEqual(comptime_float, @TypeOf(any.b));
        try std.testing.expectEqual(3.0, any.b);
        try std.testing.expectEqual(comptime_float, @TypeOf(any.a));
        try std.testing.expectEqual(4.0, any.a);
        try std.testing.expectEqual(input, any.to(@TypeOf(input)));
    }

    // Struct mixed
    {
        const input = .{ .r = 1.0, .b = 3.0 };
        const any = anyColor(input);
        try std.testing.expectEqual(comptime_float, @TypeOf(any.r));
        try std.testing.expectEqual(1.0, any.r);
        try std.testing.expectEqual(void, @TypeOf(any.g));
        try std.testing.expectEqual(comptime_float, @TypeOf(any.b));
        try std.testing.expectEqual(3.0, any.b);
        try std.testing.expectEqual(void, @TypeOf(any.a));
        try std.testing.expectEqual(input, any.to(@TypeOf(input)));
        try std.testing.expectEqual(1.0, any.to(struct { r: f32 }).r);
        try std.testing.expectEqual(3.0, any.to(struct { b: f32 }).b);
    }
}

test "specific color" {
    const srgb: [4]f32 = .{ 0.063, 0.486, 0.769, 1.0 };
    const linear: [4]f32 = .{ 0.005208, 0.2013, 0.5526, 1.0 };
    const unorm: [4]u8 = .{ 1, 51, 141, 255 };

    for (0..3) |i| try std.testing.expectApproxEqRel(linear[i], srgbToLinear([4]f32, srgb)[i], 0.0001);
    try std.testing.expectEqual(linear[3], srgbToLinear([4]f32, srgb)[3]);

    for (0..4) |i| try std.testing.expectEqual(unorm[i], floatToUnorm([4]u8, linear)[i]);

    const res = anyColor(.{ .r = 0.063, .g = 0.486, .b = 0.769, .a = 1.0 });
    try std.testing.expectEqual(0.063, res.r);
    try std.testing.expectEqual(0.486, res.g);
    try std.testing.expectEqual(0.769, res.b);
    try std.testing.expectEqual(1.0, res.a);
}
