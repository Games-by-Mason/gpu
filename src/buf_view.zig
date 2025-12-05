//! See `View`.

const std = @import("std");
const assert = std.debug.assert;

const Writer = @import("Writer.zig");

/// A view into a buffer.
pub fn BufView(Buf: type) type {
    return struct {
        const Self = @This();

        /// A handle to the buffer.
        handle: @FieldType(Buf, "handle"),
        /// A memory mapped pointer to the buffer, or void if not mapped.
        ptr: if (@hasField(Buf, "data"))
            @TypeOf(@as(@FieldType(Buf, "data"), undefined).ptr)
        else
            void,
        /// An offset into the buffer.
        offset: u64 = 0,
        /// The length of the view, starting at the offset.
        len: u64 = 0,

        pub inline fn as(
            self: @This(),
            comptime result_kind: @TypeOf(Buf.kind),
        ) BufView(@TypeOf(@as(Buf, undefined).as(result_kind))) {
            return .{
                .handle = self.handle.as(result_kind),
                .ptr = self.ptr,
                .offset = self.offset,
                .len = self.len,
            };
        }

        pub inline fn asBuf(
            self: @This(),
            comptime result_kind: @TypeOf(Buf.kind),
        ) BufView(@TypeOf(@as(Buf, undefined).asBuf(result_kind))) {
            return .{
                .handle = self.handle.as(result_kind),
                .ptr = {},
                .offset = self.offset,
                .len = self.len,
            };
        }

        pub fn writer(self: @This()) Writer {
            return .{
                .ptr = @ptrFromInt(@intFromPtr(self.ptr) + self.offset),
                .pos = 0,
                .size = self.len,
            };
        }

        pub fn spliced(self: Self, start: u64, maybe_len: ?u64) @This() {
            var result = self;
            result.splice(start, maybe_len);
            return result;
        }

        pub fn splice(self: *Self, start: u64, maybe_len: ?u64) void {
            if (start > self.len) @panic("OOB");
            self.offset += start;
            if (maybe_len) |l| {
                if (l > self.len) @panic("OOB");
                if (start > self.len - l) @panic("OOB");
                self.len = l;
            } else {
                self.len = self.len - start;
            }
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

            view: Self,
            target_segment_bytes: usize,
            min_segment_bytes: usize,

            pub fn next(self: *@This()) ?Self {
                // Check if there's any space remaining
                if (self.view.len == 0) return null;

                // Calculate the segment size
                const segment_size = if (self.view.len >= self.target_segment_bytes + self.min_segment_bytes) b: {
                    // We have enough space to return a fully sized segment
                    break :b self.target_segment_bytes;
                } else b: {
                    // We don't have enough space to return a fully sized segment, or doing so would
                    // leave us with a remainder that's under out limit. Just return all remaining
                    // bytes.
                    break :b self.view.len;
                };

                // Create the segment writer
                const result: Self = .{
                    .handle = self.view.handle,
                    .ptr = self.view.ptr,
                    .offset = self.view.offset,
                    .len = segment_size,
                };

                // Update our offset, and return the result
                self.view.len -= segment_size;
                self.view.offset += segment_size;
                return result;
            }
        };

        /// Splits this view into view of contiguous segments.
        ///
        /// Returned splits will be equivalent to the original view with the offset and size
        /// adjusted.
        pub fn splitIter(self: Self, options: SplitIterator.Options) SplitIterator {
            assert(options.min_segment_bytes > 0);
            assert(options.max_segments > 0);

            const target_segment_bytes = std.math.divCeil(
                usize,
                self.len,
                options.max_segments,
            ) catch |err| @panic(@errorName(err));

            return .{
                .view = self,
                .target_segment_bytes = target_segment_bytes,
                .min_segment_bytes = options.min_segment_bytes,
            };
        }
    };
}
