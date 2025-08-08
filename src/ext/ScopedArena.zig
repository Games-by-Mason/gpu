//! A scoped arena. Automatically resets when all open scopes end.

const std = @import("std");
const assert = std.debug.assert;
const safety = std.debug.runtime_safety;
const Allocator = std.mem.Allocator;
const Arena = std.heap.ArenaAllocator;

depth: u32,
arena: Arena,

pub fn init(child: Allocator) @This() {
    return .{
        .depth = 0,
        .arena = .init(child),
    };
}

pub fn deinit(self: *@This()) void {
    assert(self.depth == 0);
    self.arena.deinit();
}

pub fn begin(self: *@This()) Allocator {
    self.depth += 1;
    return self.arena.allocator();
}

pub fn end(self: *@This()) void {
    self.depth -= 1;
    if (self.depth == 0) _ = self.arena.reset(.retain_capacity);
}
