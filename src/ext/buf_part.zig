//! See `bufPart`.

const std = @import("std");
const gpu = @import("../root.zig");

const assert = std.debug.assert;

const Gx = gpu.Gx;
const BufView = gpu.BufView;
const DebugName = gpu.DebugName;

pub fn Options(Buf: type) type {
    return struct {
        pub const GlobalPart = struct {
            result: *Buf.View,
            size: u64,
            alignment: u16,

            pub fn init(
                T: type,
                result: *Buf.View,
            ) @This() {
                return .{
                    .result = result,
                    .size = @sizeOf(T),
                    .alignment = @alignOf(T),
                };
            }
        };

        pub const FramePart = struct {
            result: *[gpu.global_options.max_frames_in_flight]Buf.View,
            size: u64,
            alignment: u16,

            pub fn init(
                T: type,
                result: *[gpu.global_options.max_frames_in_flight]Buf.View,
            ) @This() {
                return .{
                    .result = result,
                    .size = @sizeOf(T),
                    .alignment = @alignOf(T),
                };
            }
        };

        const BufOptions = if (@hasField(Buf.Options, "prefer_device_local")) struct {
            name: DebugName,
            prefer_device_local: bool,

            fn asBufOptions(self: @This(), size: u64) Buf.Options {
                return .{
                    .name = self.name,
                    .size = size,
                    .prefer_device_local = self.prefer_device_local,
                };
            }
        } else struct {
            name: DebugName,
            size: u64,

            fn asBufOptions(self: @This(), size: u64) Buf.Options {
                return .{
                    .name = self.name,
                    .size = size,
                };
            }
        };

        buf: BufOptions,
        global: []const GlobalPart = &.{},
        frame: []const FramePart = &.{},
    };
}

/// A suballocator for buffers. For managing image allocations, see `gpu.ext.ImageBumpAllocator`.
///
/// Modern graphics APIs provide what is essentially a page allocator, and expect you to suballocate
/// from it. However, different devices may require different alignment, which can make this tricky
/// to manage.
///
/// These APIs typically allow you to bind multiple buffers to the same backing memory. However,
/// vendors often recommend suballocating buffers yourself instead.
///
/// `bufPart` allocates a single backing buffer, and then suballocates it into view for you.
/// Alignment is handled automatically.
///
/// Partitions can either be global, or per frame. A global partition gets a single instance,
/// whereas a frame based partition gets one view per frame in flight.
///
/// Buffers are laid out in the following order:
/// 1. Global partitions, in declared order
/// 2. Frame 0 partitions, in declared order
/// 3. Frame 1 partitions, in declared order
/// 4. ...
///
/// For best performance across all hardware, order frequently updated partitions in the order that
/// you'll write to them to minimize seeking in write combined memory.
pub fn bufPart(gx: *Gx, Buf: type, options: Options(Buf)) Buf {
    // Calculate the minimum alignment for this buffer type
    const buf_align = b: {
        comptime assert(std.meta.fields(gpu.BufKind).len == 8); // Handle all of them!
        var buf_align: u16 = 1;
        if (Buf.kind.transfer_src or Buf.kind.transfer_dst) {
            buf_align = @max(buf_align, gpu.Device.buffer_copy_offset_alignment);
        }
        if (Buf.kind.uniform_texel or Buf.kind.storage_texel) {
            buf_align = @max(buf_align, gx.device.texel_buffer_offset_alignment);
        }
        if (Buf.kind.uniform) {
            buf_align = @max(buf_align, gx.device.uniform_buf_offset_alignment);
        }
        if (Buf.kind.storage) {
            buf_align = @max(buf_align, gx.device.storage_buf_offset_alignment);
        }
        // The alignment requirements for these doesn't appear to be documented. It may be
        // the case that only the offsets into them need to be properly aligned. Regardless,
        // I've set the minimum alignment to the std140 alignment of the underlying types
        // just in case.
        if (Buf.kind.indirect) {
            buf_align = @max(buf_align, 4);
        }
        if (Buf.kind.index) {
            buf_align = @max(buf_align, 2);
        }
        break :b buf_align;
    };

    // Calculate the offset and size for each part, and the total size, factoring in both the
    // buffer alignment and the alignment requirements of each part's contents
    const size = b: {
        var offset: u64 = 0;
        for (options.global) |part| {
            offset = std.mem.alignForward(u64, offset, @max(buf_align, part.alignment));
            part.result.* = .{
                .offset = offset,
                .len = part.size,
                .ptr = undefined,
                .handle = undefined,
            };
            offset += part.size;
        }

        for (0..gpu.global_options.max_frames_in_flight) |frame| {
            for (options.frame) |part| {
                offset = std.mem.alignForward(u64, offset, @max(buf_align, part.alignment));
                part.result[frame].offset = offset;
                part.result[frame] = .{
                    .offset = offset,
                    .len = part.size,
                    .ptr = undefined,
                    .handle = undefined,
                };
                offset += part.size;
            }
        }
        break :b offset;
    };

    // Create a buffer big enough to hold all the parts
    const buf: Buf = .init(gx, options.buf.asBufOptions(size));

    // Initialize the handle and data pointers and then return the buffer
    for (options.global) |part| {
        part.result.handle = buf.handle;
        part.result.ptr = if (@hasField(Buf, "data")) buf.data.ptr else {};
    }
    for (options.frame) |part| {
        for (0..gpu.global_options.max_frames_in_flight) |frame| {
            part.result[frame].handle = buf.handle;
            part.result[frame].ptr = if (@hasField(Buf, "data")) buf.data.ptr else {};
        }
    }

    return buf;
}
