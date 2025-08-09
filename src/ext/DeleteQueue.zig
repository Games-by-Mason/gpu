//! In modern graphics APIs, you often have to wait until a resource is no longer in use by the GPU
//! to free it. Often this requires waiting until the frame in flight counter loops. This structure
//! makes this a bit easier.
//!
//! Your renderer will typically store one delete queue per frame in flight, and reset it at the
//! beginning of every frame. If you want to destroy a resource once the frame in flight loops,
//! you can just enqueue it to this queue.

const std = @import("std");
const gpu = @import("../root.zig");

const assert = std.debug.assert;

const Gx = gpu.Gx;
const Allocator = std.mem.Allocator;

/// Represents a handle queued for destruction.
pub const Handle = struct {
    /// The integer value of the handle.
    value: u64,
    /// A pointer to the handle's delete function.
    deinit: *const fn (value: u64, gx: *Gx) void,
};

/// The list of handles queued for deletion.
handles: std.ArrayListUnmanaged(Handle) = .{},
/// If more than `1/warn_ratio` of the storage is used, a warning will be emitted. Zero disables the
/// warning.
warn_ratio: u8 = 4,

pub fn initCapacity(gpa: Allocator, capacity: usize) Allocator.Error!@This() {
    return .{ .handles = try std.ArrayListUnmanaged(Handle).initCapacity(gpa, capacity) };
}

pub fn deinit(self: *@This(), gpa: Allocator, gx: *Gx) void {
    self.reset(gx);
    self.handles.deinit(gpa);
}

/// Appends the given resource to the delete queue.
///
/// Checks for various relevant fields on GPU types.
pub fn append(self: *@This(), resource: anytype) void {
    switch (@typeInfo(@TypeOf(resource))) {
        .@"enum" => {
            comptime assert(@sizeOf(@TypeOf(resource)) == @sizeOf(u64));
            const Deinit = struct {
                fn deinit(value: u64, gx: *Gx) void {
                    const handle: @TypeOf(resource) = @enumFromInt(value);
                    handle.deinit(gx);
                }
            };
            self.handles.appendBounded(.{
                .value = @intFromEnum(resource),
                .deinit = &Deinit.deinit,
            }) catch @panic("OOB");
        },
        .@"struct" => {
            self.append(resource.handle);
            if (@hasField(@TypeOf(resource), "view")) {
                self.append(resource.view);
            }
            if (@hasField(@TypeOf(resource), "memory")) {
                self.append(resource.memory);
            }
        },
        else => comptime unreachable,
    }
}

/// Frees all queues resources and resets the queue.
pub fn reset(self: *@This(), gx: *Gx) void {
    if (self.warn_ratio != 0 and self.handles.items.len > self.handles.capacity / self.warn_ratio) {
        std.log.warn(
            "delete queue {x} past 1/{} capacity",
            .{ @intFromPtr(self), self.warn_ratio },
        );
    }
    for (self.handles.items) |handle| {
        handle.deinit(handle.value, gx);
    }
    self.handles.clearRetainingCapacity();
}
