//! A writer to volatile write-only memory.

const std = @import("std");
const assert = std.debug.assert;

pub const Error = error{Overflow};
const Self = @This();
const Dest = [*]volatile u8;

ptr: *volatile anyopaque,
pos: u64 = 0,
size: u64 = 0,

pub fn initSlice(from: anytype) Self {
    comptime assert(@typeInfo(@TypeOf(from)).Pointer.size == .Slice);
    const bytes = std.mem.sliceAsBytes(from);
    return .{
        .ptr = bytes.ptr,
        .size = bytes.len,
        .pos = 0,
    };
}

pub inline fn write(self: *Self, bytes: []const u8) Error!usize {
    try self.writeAll(bytes);
    return bytes.len;
}

pub inline fn writeAll(self: *Self, bytes: []const u8) Error!void {
    // Copy memory
    const src = bytes[0..@min(bytes.len, self.remainingBytes())];
    const dest: Dest = @ptrFromInt(@intFromPtr(self.ptr) + self.pos);
    @memcpy(dest, src);

    // Update pos
    self.pos += bytes.len;

    // Return an error if we overflowed the buffer
    if (src.len < bytes.len) return Error.Overflow;
}

pub inline fn print(self: *Self, comptime format: []const u8, args: anytype) Error!void {
    return @errorCast(self.any().print(format, args));
}

pub inline fn writeByte(self: *Self, byte: u8) Error!void {
    return @errorCast(self.any().writeByte(byte));
}

pub inline fn writeByteNTimes(self: *Self, byte: u8, n: usize) Error!void {
    return @errorCast(self.any().writeByteNTimes(byte, n));
}

pub inline fn writeBytesNTimes(self: *Self, bytes: []const u8, n: usize) Error!void {
    return @errorCast(self.any().writeBytesNTimes(bytes, n));
}

pub inline fn writeStructAligned(self: *Self, value: anytype) Error!void {
    try self.alignForward(@alignOf(@TypeOf(value)));
    try self.writeStruct(value);
}

pub inline fn writeStructAssumeAligned(self: *Self, value: anytype) Error!void {
    assert(self.pos % @alignOf(@TypeOf(value)) == 0);
    try self.writeStruct(value);
}

/// Always writes native endian. In practice, this is what you want when writing data for the GPU
/// to read.
///
/// https://docs.vulkan.org/spec/latest/chapters/fundamentals.html#fundamentals-host-environment
pub inline fn writeStruct(self: *Self, value: anytype) Error!void {
    // Implementation manually inlined to improve debug mode performance
    comptime assert(@typeInfo(@TypeOf(value)).@"struct".layout != .auto);
    return self.writeAll(std.mem.asBytes(&value));
}

pub inline fn any(self: *Self) std.io.AnyWriter {
    return .{
        .context = @ptrCast(&self),
        .writeFn = typeErasedWriteFn,
    };
}

fn typeErasedWriteFn(context: *const anyopaque, bytes: []const u8) anyerror!usize {
    const ptr: *const *Self = @alignCast(@ptrCast(context));
    return write(ptr.*, bytes);
}

pub fn seekTo(self: *Self, pos: u64) void {
    // Check bounds
    if (pos > self.size - self.pos) @panic("OOB");

    // Set the pos
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
pub fn alignForward(self: *Self, comptime alignment: u16) Error!void {
    const max_padding = [_]u8{0} ** alignment;
    const len = std.mem.alignForward(u64, self.pos, alignment) - self.pos;
    try self.writeAll(max_padding[0..len]);
}

pub fn remainingBytes(self: *const Self) usize {
    return self.size - self.pos;
}
