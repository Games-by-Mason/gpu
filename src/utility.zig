const std = @import("std");
const assert = std.debug.assert;

pub const log = std.log.scoped(.gpu);

/// Assumes a buffer is null terminated, and returns the string it contains. If it turns out not to
/// be null terminated, the whole buffer is returned.
pub fn bufToStr(buf: anytype) []const u8 {
    comptime assert(@typeInfo(@TypeOf(buf)) == .pointer);
    for (buf, 0..) |c, i| {
        if (c == 0) return buf[0..i];
    }
    return buf;
}

pub fn containsBits(self: anytype, other: @TypeOf(self)) bool {
    const Int = std.meta.Int(.unsigned, @bitSizeOf(@TypeOf(self)));
    const self_bits: Int = @bitCast(self);
    const other_bits: Int = @bitCast(other);
    return self_bits & other_bits == other_bits;
}
