//! See `View`.

const std = @import("std");
const assert = std.debug.assert;

pub const Options = struct {
    Handle: type,
    Ptr: type,
};

/// A view into a buffer.
pub fn View(opt: Options) type {
    return struct {
        const Self = @This();

        /// A handle to the buffer.
        handle: opt.Handle,
        /// A memory mapped pointer to the buffer, or void if not mapped.
        ptr: opt.Ptr,
        /// An offset into the buffer.
        offset: u64 = 0,
        /// The total size of the buffer, not factoring in the offset.
        buf_size: u64 = 0,

        /// Returns the length of this view.
        pub fn len(self: *const @This()) u64 {
            return self.buf_size - self.offset;
        }

        pub fn spliced(self: Self, start: u64, maybe_len: ?u64) @This() {
            var result = self;
            result.splice(start, maybe_len);
            return result;
        }

        pub fn splice(self: *Self, start: u64, maybe_len: ?u64) void {
            if (self.offset > self.buf_size) @panic("OOB");
            const remaining = self.buf_size - self.offset;
            if (start > remaining) @panic("OOB");
            self.offset += start;
            if (maybe_len) |l| {
                if (l > remaining - start) @panic("OOB");
                self.buf_size = self.offset + l;
            } else {
                self.buf_size = self.buf_size - start;
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
                const remaining = self.view.len();
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
                    .handle = self.view.handle,
                    .ptr = self.view.ptr,
                    .offset = self.view.offset,
                    .buf_size = self.view.offset + segment_size,
                };

                // Update our offset, and return the result
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
                self.buf_size - self.offset,
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
