//! A scoped arena backed by a fixed buffer allocator.

const std = @import("std");
const assert = std.debug.assert;
const safety = std.debug.runtime_safety;
const Allocator = std.mem.Allocator;
const FixedBufferAllocator = std.heap.FixedBufferAllocator;

/// The backing allocator.
fba: FixedBufferAllocator,
/// The start of this scope.
watermark: usize,
/// If more than to `1/warn_ratio` of the storage is used, a warning will be emitted. Zero disables
/// the warning.
warn_ratio: u8 = 4,

/// We over-align the buffer so that initial alignment is unlikely to affect the effective capacity.
const buf_align: std.mem.Alignment = .of(usize);

/// Initializes the arena with a fixed capacity
pub fn init(gpa: Allocator, capacity_log2: usize) Allocator.Error!@This() {
    const buf = try gpa.alignedAlloc(u8, buf_align, std.math.pow(usize, 2, capacity_log2));
    errdefer comptime unreachable;

    return .{
        .fba = .init(buf),
        .watermark = 0,
    };
}

/// Destroys the arena, asserting that all scopes were popped.
pub fn deinit(self: *@This(), gpa: Allocator) void {
    assert(self.watermark == 0);
    gpa.free(@as([]align(buf_align.toByteUnits()) u8, @alignCast(self.fba.buffer)));
    self.* = undefined;
}

/// Begins a new scope.
pub fn begin(self: *@This()) Allocator.Error!Allocator {
    const allocator = self.fba.allocator();

    const next_watermark = self.fba.end_index;
    const prev_watermark = try allocator.create(usize);

    prev_watermark.* = self.watermark;
    self.watermark = next_watermark;

    return allocator;
}

/// Ends the current scope.
pub fn end(self: *@This()) void {
    assert(self.fba.end_index > self.watermark);

    if (self.warn_ratio != 0 and self.fba.end_index >= self.fba.buffer.len / self.warn_ratio) {
        std.log.warn(
            "scoped arena {x} past 1/{} capacity",
            .{ @intFromPtr(self), self.warn_ratio },
        );
    }

    const freed = self.fba.buffer[self.watermark..self.fba.end_index];

    self.fba.end_index = self.watermark;
    const prev_watermark: *usize = @ptrFromInt(std.mem.alignForward(
        usize,
        @intFromPtr(&self.fba.buffer[self.fba.end_index]),
        @alignOf(usize),
    ));
    self.watermark = prev_watermark.*;

    @memset(freed, undefined);
}

test "all" {
    const ScopedArena = @This();

    var arena = try ScopedArena.init(std.testing.allocator, 6);
    arena.warn_ratio = 0;
    defer arena.deinit(std.testing.allocator);

    for (0..2) |_| {
        const scope_0_restore = arena.fba.end_index;
        try std.testing.expectEqual(scope_0_restore, 0);

        const scope_0 = try arena.begin();

        const a0 = try scope_0.create(u8);
        a0.* = 0;
        const a1 = try scope_0.create(u16);
        a1.* = 1;

        {
            const scope_1_restore = arena.fba.end_index;
            const scope_1 = try arena.begin();

            const a2 = try scope_1.create(u32);
            a2.* = 2;
            const a3 = try scope_1.create(u64);
            a3.* = 3;

            try std.testing.expectEqual(2, a2.*);
            try std.testing.expectEqual(3, a3.*);

            arena.end();

            try std.testing.expectEqual(scope_1_restore, arena.fba.end_index);
        }

        {
            const scope_1_restore = arena.fba.end_index;
            _ = try arena.begin();
            arena.end();
            try std.testing.expectEqual(scope_1_restore, arena.fba.end_index);
        }

        {
            const scope_1_restore = arena.fba.end_index;
            const scope_1 = try arena.begin();

            const a4 = try scope_1.create(u8);
            a4.* = 4;
            const a5 = try scope_1.create(u32);
            a5.* = 5;

            try std.testing.expectEqual(4, a4.*);
            try std.testing.expectEqual(5, a5.*);

            {
                const scope_2_restore = arena.fba.end_index;
                _ = try arena.begin();

                const scope_3_restore = arena.fba.end_index;
                const scope_3 = try arena.begin();

                const a6 = try scope_3.create(u4);
                a6.* = 6;
                const a7 = try scope_3.create(u4);
                a7.* = 7;

                try std.testing.expectError(error.OutOfMemory, scope_3.create(u256));

                try std.testing.expectEqual(6, a6.*);
                try std.testing.expectEqual(7, a7.*);

                arena.end();

                try std.testing.expectEqual(scope_3_restore, arena.fba.end_index);

                arena.end();

                try std.testing.expectEqual(scope_2_restore, arena.fba.end_index);
            }

            arena.end();

            try std.testing.expectEqual(scope_1_restore, arena.fba.end_index);
        }

        try std.testing.expectEqual(0, a0.*);
        try std.testing.expectEqual(1, a1.*);

        arena.end();

        try std.testing.expectEqual(scope_0_restore, arena.fba.end_index);
    }

    try std.testing.expectEqual(0, arena.fba.end_index);
}
