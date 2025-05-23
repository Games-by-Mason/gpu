//! A writer to volatile write-only memory.

const std = @import("std");
const assert = std.debug.assert;

const Self = @This();
const Dest = [*]volatile u8;

ptr: *volatile anyopaque,
pos: u64,
size: u64,

/// Similar to `Writer`, but can only write type `T`.
pub fn Typed(T: type) type {
    return struct {
        ptr: *volatile anyopaque,
        capacity: u64,
        len: u64,

        pub fn write(self: *@This(), value: T) void {
            comptime assert(@typeInfo(T).@"struct".layout != .auto);
            const offset = std.math.mul(u64, self.len, @sizeOf(T)) catch @panic("OOB");
            if (offset >= self.capacity * @sizeOf(T)) @panic("OOB");
            const dest: *align(1) volatile T = @ptrFromInt(@intFromPtr(self.ptr) + offset);
            dest.* = value;
            self.len += 1;
        }
    };
}

/// Returns a typed writer of the given type. Asserts that writer is properly aligned for the type.
pub fn typed(self: @This(), T: type) Typed(T) {
    assert(self.pos % @alignOf(T) == 0);
    return .{
        .ptr = @ptrFromInt(@intFromPtr(self.ptr) + self.pos),
        .len = 0,
        .capacity = (self.size - self.pos) / @sizeOf(T),
    };
}

/// Writes the given bytes to the writer, panicking if it goes out of bounds.
pub inline fn writeAll(self: *Self, bytes: []const u8) void {
    const new_pos = std.math.add(u64, self.pos, bytes.len) catch @panic("OOB");
    if (new_pos > self.size) @panic("OOB");
    const src = bytes[0..@min(bytes.len, self.remainingBytes())];
    const dest: Dest = @ptrFromInt(@intFromPtr(self.ptr) + self.pos);
    @memcpy(dest, src);
    self.pos = new_pos;
}

/// Like `writeStructUnaligned`, but asserts that the writer is aligned to the alignment of the
/// struct.
pub inline fn writeStruct(self: *Self, value: anytype) void {
    assert(self.pos % @alignOf(@TypeOf(value)) == 0);
    self.writeStructUnaligned(value);
}

/// Like `writeStructUnaligned`, but aligns the writer before writing.
pub inline fn writeStructAligned(self: *Self, value: anytype) void {
    self.alignForward(@alignOf(@TypeOf(value)));
    self.writeStructUnaligned(value);
}

/// Writes a struct to the writer, ignoring alignment.
///
/// Always writes native endian. In practice, this is what you want when writing data for the GPU to
/// read.
///
/// https://docs.vulkan.org/spec/latest/chapters/fundamentals.html#fundamentals-host-environment
pub inline fn writeStructUnaligned(self: *Self, value: anytype) void {
    comptime assert(@typeInfo(@TypeOf(value)).@"struct".layout != .auto);
    const new_pos = std.math.add(u64, self.pos, @sizeOf(@TypeOf(value))) catch @panic("OOB");
    if (new_pos > self.size) @panic("OOB");
    const dest: *align(1) volatile @TypeOf(value) = @ptrFromInt(@intFromPtr(self.ptr) + self.pos);
    dest.* = value;
    self.pos = new_pos;
}

/// Returns a `std.io.AnyWriter`. Will panic if used to write out of bounds.
pub inline fn any(self: *Self) std.io.AnyWriter {
    return .{
        .context = @ptrCast(&self),
        .writeFn = typeErasedWriteFn,
    };
}

fn typeErasedWriteFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
    const ptr: *const *Self = @alignCast(@ptrCast(context));
    writeAll(ptr.*, bytes);
    return bytes.len;
}

pub fn seekTo(self: *Self, pos: u64) void {
    if (pos > self.size - self.pos) @panic("OOB");
    self.pos = pos;
}

pub fn spliced(self: Self, start: u64, maybe_len: ?u64) @This() {
    var result = self;
    result.splice(start, maybe_len);
    return result;
}

pub fn splice(self: *Self, start: u64, maybe_len: ?u64) void {
    if (self.pos > self.size) @panic("OOB");
    const remaining = self.size - self.pos;
    if (start > remaining) @panic("OOB");
    self.pos += start;
    if (maybe_len) |len| {
        if (len > remaining - start) @panic("OOB");
        self.size = self.pos + len;
    } else {
        self.size = self.size - start;
    }
}

// Pads with `0`s to get the desired alignment. This is expected to be more efficient than
// seeking, as seeking may cause unnecessary flushes of write combined memory on some
// hardware.
pub fn alignForward(self: *Self, comptime alignment: u16) void {
    const max_padding = [_]u8{0} ** alignment;
    const len = std.mem.alignForward(u64, self.pos, alignment) - self.pos;
    self.writeAll(max_padding[0..len]);
}

pub fn remainingBytes(self: *const Self) usize {
    return self.size - self.pos;
}
