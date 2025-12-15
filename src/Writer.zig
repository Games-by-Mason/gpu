//! A writer for memory mapped GPU buffers.
//!
//! On typical hardware, memory mapped GPU buffers are write combined, and they are often configured
//! by the user to be uncached. This results in two common performance pitfalls:
//!
//! 1. Reading from uncached memory. This is slow since the memory is uncached, you're not expected
//!    to be reading from it on the CPU in practice. You could make this mistake because you're
//!    simply unaware of this tradeoff, but you could also get this wrong in more subtle ways--e.g.
//!    by assigning a packed type to uncached memory, or by inadvertent incrementing a value
//!    rather than assigning to it.
//!
//! 2. Writing to write combined memory out of order. On typical hardware, these buffers are write
//!    combined, which means that every N bytes you write will be flushed to the GPU all at once.
//!    This works best when you fill the buffer in order from start to finish. The more you jump
//!    around, the greater the chance of incurring unnecessary flushes.
//!
//! These two factors make a writer the perfect interface for uploading data to GPU buffers: you
//! can't read from a writer, and you can't write out of order. Anyopaque is used as the pointer
//! field to decrease likelihood of inadvertently reading this data, e.g. in a log or such.
//!
//! For maximum performance, you can call the lower level unbuffered non virtual methods provided
//! directly on this type. These methods assert on failure, since in practice, you're unlikely to be
//! able to or want to handle these writes failing. These types assume native byte order unless
//! otherwise specified, this is almost certainly what you want in practice:
//!
//! * https://docs.vulkan.org/spec/latest/chapters/fundamentals.html#fundamentals-host-environment
//!
//! For maximum interoperability--for example, streaming files from disk to the GPU using std.Io--
//! use the provided IO writer interface. Keep in mind that if you buffer any data through the IO
//! writer interface, you likely want to flush it before using the unbuffered interface.
//!
//! You should typically set the buffer size to zero. The only common exception is when you're
//! interleaving writes to multiple buffers, in this case you may want to consider a larger buffer
//! size to reduce unnecessary flushing to the GPU. However, your target hardware may be able to
//! handle filling a few buffers simultaneously. When in doubt, profile.

const std = @import("std");
const assert = std.debug.assert;

const Self = @This();
const Io = std.Io;

ptr: *anyopaque,
pos: u64,
size: u64,
interface: Io.Writer,

pub const Options = struct {
    ptr: *anyopaque,
    pos: u64,
    size: u64,
    buffer: []u8,
};

pub fn init(options: Options) @This() {
    return .{
        .ptr = options.ptr,
        .pos = options.pos,
        .size = options.size,
        .interface = initInterface(options.buffer),
    };
}

pub fn initInterface(buffer: []u8) Io.Writer {
    return .{
        .vtable = &.{
            .drain = &drain,
            .sendFile = &sendFile,
        },
        .buffer = buffer,
    };
}

/// Writes bytes immediately, bypassing the buffer. Asserts on OOB.
pub fn writeAllUnbuffered(self: *Self, bytes: []const u8) void {
    const ptr: [*]u8 = @ptrCast(self.ptr);
    const dst = ptr[0..self.size][self.pos..][0..bytes.len];
    @memcpy(dst, bytes);

    self.pos += bytes.len;
    assert(self.pos <= self.size);
}

/// Writes the given struct immediately with native byte order, bypassing the buffer and prefixing
/// the struct with any padding required by the alignment. Asserts on OOB.
pub fn writeStructAlignedUnbuffered(self: *Self, value: anytype) void {
    self.alignForward(@alignOf(@TypeOf(value)));
    writeStructUnbufferedInner(self, value, @alignOf(@TypeOf(value)));
}

/// Writes the given struct immediately with native byte order, bypassing the buffer. No alignment
/// requirements. Asserts on OOB.
pub fn writeStructUnalignedUnbuffered(self: *Self, value: anytype) void {
    writeStructUnbufferedInner(self, value, 1);
}

/// Writes the given struct immediately with native byte order, bypassing the buffer. Asserts on OOB
/// or misaligned writes.
pub fn writeStructUnbuffered(self: *Self, value: anytype) void {
    writeStructUnbufferedInner(self, value, @alignOf(@TypeOf(value)));
}

fn writeStructUnbufferedInner(self: *Self, value: anytype, alignment: comptime_int) void {
    comptime assert(@typeInfo(@TypeOf(value)).@"struct".layout != .auto);
    const dest: *align(alignment) @TypeOf(value) = @ptrCast(@alignCast(self.dst + self.pos));
    dest.* = value;

    self.pos += @sizeOf(value);
    assert(self.pos <= self.size);
}

/// Seeks to the given position. Asserts on OOB.
pub fn seekTo(self: *Self, pos: u64) void {
    self.interface.flush() catch unreachable;
    self.pos = pos;
    assert(self.pos <= self.size);
}

/// Returns the splices writer with a new buffer. Asserts on OOB.
pub fn spliced(self: Self, start: u64, maybe_len: ?u64, buffer: []u8) @This() {
    var result = self;
    result.splice(start, maybe_len);
    result.interface.buffer = buffer;
    return result;
}

/// Splices the writer. Asserts on OOB.
pub fn splice(self: *Self, start: u64, maybe_len: ?u64) void {
    self.interface.flush() catch unreachable;

    self.pos += start;
    assert(self.pos <= self.size);

    if (maybe_len) |len| {
        assert(self.size - self.pos <= len);
        self.size = self.pos + len;
    } else {
        self.size = self.size - start;
    }
}

// Immediately pads with `0`s to get the desired alignment, bypassing the buffer. This is expected
// to be more efficient than seeking, as seeking may cause unnecessary flushes of write combined
// memory on typical hardware.
pub fn alignForwardUnbuffered(self: *Self, comptime alignment: u16) void {
    const max_padding = [_]u8{0} ** alignment;
    const len = std.mem.alignForward(u64, self.pos, alignment) - self.pos;
    self.writeAllUnbuffered(max_padding[0..len]);
}

/// Returns the number of bytes left in this writer.
pub fn remainingBytes(self: *const Self) usize {
    return self.size - self.pos - self.interface.buffer.len;
}

/// Similar to `Writer`, but unbuffered and can only write type `T`.
pub fn Typed(T: type) type {
    _ = extern struct { extern_type: T };
    return struct {
        ptr: *anyopaque,
        capacity: u64,
        len: u64,

        pub fn write(self: *@This(), value: T) void {
            const offset = std.math.mul(u64, self.len, @sizeOf(T)) catch @panic("OOB");
            if (offset >= self.capacity * @sizeOf(T)) @panic("OOB");
            const ptr: [*]u8 = @ptrCast(@alignCast(self.ptr));
            const dest_untyped = &ptr[offset];
            const dest: *T = @ptrCast(@alignCast(dest_untyped));
            dest.* = value;
            self.len += 1;
        }
    };
}

/// Returns a typed writer of the given type. Asserts that writer is properly aligned for the type.
pub fn typed(self: @This(), T: type) Typed(T) {
    const ptr: [*]u8 = @ptrCast(self.ptr);
    const dest_untyped = &ptr[self.pos];
    const dest: *T = @ptrCast(@alignCast(dest_untyped));
    return .{
        .ptr = @ptrCast(dest),
        .len = 0,
        .capacity = (self.size - self.pos) / @sizeOf(T),
    };
}

fn drain(w: *Io.Writer, data: []const []const u8, splat: usize) Io.Writer.Error!usize {
    const self: *@This() = @fieldParentPtr("interface", w);

    // Flush the buffer
    self.writeAllUnbuffered(w.buffered());
    w.end = 0;

    // Write the initial vecs
    var consumed: usize = 0;
    for (data[0 .. data.len - 1]) |v| {
        self.writeAllUnbuffered(v);
        consumed += v.len;
    }

    // Splat the last vec
    const last = data[data.len - 1];
    for (0..splat) |_| {
        self.writeAllUnbuffered(last);
    }
    consumed += last.len * splat;

    return consumed;
}

fn sendFile(
    w: *Io.Writer,
    file_reader: *Io.File.Reader,
    limit: Io.Limit,
) Io.Writer.FileError!usize {
    const self: *@This() = @fieldParentPtr("interface", w);

    // Flush the buffer
    self.writeAllUnbuffered(w.buffered());
    w.end = 0;

    // Check for end of stream
    if (file_reader.atEnd()) return error.EndOfStream;

    // Check remaining space
    const ptr: [*]u8 = @ptrCast(self.ptr);
    const remaining = ptr[0..self.size][self.pos..];
    const dest = limit.slice(remaining);
    if (dest.len == 0) return error.WriteFailed;

    // Read the file
    const consumed = try file_reader.interface.readSliceShort(dest);
    self.pos += consumed;
    return consumed;
}

test {
    std.testing.refAllDecls(@This());
}
