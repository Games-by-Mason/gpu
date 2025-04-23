//! Provides writers that own their write destination. Typically used for uploading memory to the
//! GPU, but may also be used by application code.

const std = @import("std");
const assert = std.debug.assert;

pub const OwnedWriterVolatile = OwnedWriterMaybeVolatile(true);
pub const OwnedWriter = OwnedWriterMaybeVolatile(false);

pub fn OwnedWriterMaybeVolatile(is_volatile: bool) type {
    return struct {
        pub const Error = error{Overflow};
        const Self = @This();
        const Dest = if (is_volatile) [*]volatile u8 else [*]u8;

        write_only_memory: *volatile anyopaque,
        pos: u64 = 0,
        size: u64 = 0,

        pub fn initSlice(from: anytype) Self {
            comptime assert(@typeInfo(@TypeOf(from)).Pointer.size == .Slice);
            const bytes = std.mem.sliceAsBytes(from);
            return .{
                .write_only_memory = bytes.ptr,
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
            const dest: Dest = @ptrFromInt(@intFromPtr(self.write_only_memory) + self.pos);
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

        pub inline fn writeInt(self: *Self, comptime T: type, value: T, endian: std.builtin.Endian) Error!void {
            return @errorCast(self.any().writeInt(T, value, endian));
        }

        pub inline fn writeStructAligned(self: *Self, value: anytype) Error!void {
            try self.alignForward(@alignOf(@TypeOf(value)));
            try self.writeStruct(value);
        }

        pub inline fn writeStructAssumeAligned(self: *Self, value: anytype) Error!void {
            assert(self.pos % @alignOf(@TypeOf(value)) == 0);
            try self.writeStruct(value);
        }

        pub inline fn writeStruct(self: *Self, value: anytype) Error!void {
            // Implementation manually inlined to improve debug mode performance
            comptime assert(@typeInfo(@TypeOf(value)).@"struct".layout != .auto);
            return self.writeAll(std.mem.asBytes(&value));
        }

        pub inline fn writeStructEndian(self: *Self, value: anytype, endian: std.builtin.Endian) Error!void {
            return @errorCast(self.any().writeStructEndian(value, endian));
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

        pub const SplitIterator = struct {
            pub const Options = struct {
                /// The iterator will try to generate this many segments, but will fall short if doing
                /// so would violate min segment bytes.
                max_segments: usize,
                /// Enforced unless the entire buffer is under this limit, in which case it's returned
                /// as a single segment.
                min_segment_bytes: usize,
            };

            writer: *Self,
            target_segment_bytes: usize,
            min_segment_bytes: usize,

            pub fn next(self: *@This()) ?Self {
                // Check if there's any space remaining
                const remaining = self.writer.remainingBytes();
                if (remaining == 0) return null;

                // Calculate the segment size
                const segment_size = if (remaining >= self.target_segment_bytes + self.min_segment_bytes) b: {
                    // We have enough space to return a fully sized segment
                    break :b self.target_segment_bytes;
                } else b: {
                    // We don't have enough space to return a fully sized segment, or doing so would
                    // leave us with a remainder that's under out limit. Just return all remaining
                    // bytes.
                    break :b remaining;
                };

                // Create the segment writer
                const result: Self = .{
                    .write_only_memory = self.writer.write_only_memory,
                    .pos = self.writer.pos,
                    .size = self.writer.pos + segment_size,
                };

                // Update our pos, and return the result
                self.writer.pos += segment_size;
                return result;
            }
        };

        /// Splits this writer into writers for contiguous segments starting at the current pos.
        ///
        /// This writer's position will be incremented for the size of each split returned.
        ///
        /// Returned splits will be equivalent to the original writer with the pos and size adjusted.
        pub fn splitIter(self: *Self, options: SplitIterator.Options) SplitIterator {
            assert(options.min_segment_bytes > 0);
            assert(options.max_segments > 0);

            const target_segment_bytes = std.math.divCeil(
                usize,
                self.size - self.pos,
                options.max_segments,
            ) catch |err| @panic(@errorName(err));

            return .{
                .writer = self,
                .target_segment_bytes = target_segment_bytes,
                .min_segment_bytes = options.min_segment_bytes,
            };
        }
    };
}
