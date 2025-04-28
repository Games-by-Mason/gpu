//! All backends must implement this interface.

const std = @import("std");
const tracy = @import("tracy");
const Allocator = std.mem.Allocator;
const Ctx = @import("Ctx.zig");

/// Creates a backend from `Self`, or emits a compiler error if `Self` does not implement this
/// interface.
pub fn IBackend(Self: type) type {
    return struct {
        Buf: type,
        CmdBuf: type,
        DescPool: type,
        DescSet: type,
        DescSetLayout: type,
        Memory: type,
        Image: type,
        ImageView: type,
        Queue: type,
        Pipeline: type,
        PipelineLayout: type,
        Sampler: type,

        InitOptions: type,

        init: fn (
            options: Ctx.InitOptionsImpl(Self.InitOptions),
        ) Self,
        deinit: fn (
            self: *Ctx,
            gpa: Allocator,
        ) void,

        dedicatedBufCreate: fn (
            self: *Ctx,
            name: Ctx.DebugName,
            kind: Ctx.BufKind,
            size: u64,
        ) Ctx.DedicatedBufResult(Ctx.DedicatedBuf(.{})),
        dedicatedUploadBufCreate: fn (
            self: *Ctx,
            name: Ctx.DebugName,
            kind: Ctx.BufKind,
            size: u64,
            prefer_device_local: bool,
        ) Ctx.DedicatedBufResult(Ctx.DedicatedUploadBuf(.{})),
        dedicatedReadbackBufCreate: fn (
            self: *Ctx,
            name: Ctx.DebugName,
            kind: Ctx.BufKind,
            size: u64,
        ) Ctx.DedicatedBufResult(Ctx.DedicatedReadbackBuf(.{})),

        bufDestroy: fn (
            self: *Ctx,
            buffer: Ctx.Buf(.{}),
        ) void,

        combinedPipelineLayoutCreate: fn (
            self: *Ctx,
            comptime max_descriptors: u32,
            options: Ctx.CombinedPipelineLayout.InitOptions,
        ) Ctx.CombinedPipelineLayout,
        combinedPipelineLayoutDestroy: fn (
            self: *Ctx,
            layout: Ctx.CombinedPipelineLayout,
        ) void,

        zoneBegin: fn (
            self: *Ctx,
            options: Ctx.Zone.BeginOptions,
        ) Ctx.Zone,
        zoneEnd: fn (
            self: *Ctx,
            cb: Ctx.CmdBuf,
        ) void,

        cmdBufGraphicsAppend: fn (
            self: *Ctx,
            cmds: Ctx.CombinedCmdBuf(null),
            options: Ctx.CombinedCmdBuf(null).AppendGraphicsCmdsOptions,
        ) void,
        cmdBufTransferAppend: fn (
            self: *Ctx,
            cmds: Ctx.CombinedCmdBuf(null),
            comptime max_regions: u32,
            options: Ctx.CombinedCmdBuf(null).AppendTransferCmdsOptions,
        ) void,

        combinedCmdBufCreate: fn (
            self: *Ctx,
            options: Ctx.CombinedCmdBufCreateOptions,
        ) Ctx.CombinedCmdBuf(null),
        combinedCmdBufSubmit: fn (
            self: *Ctx,
            combined_command_buffer: Ctx.CombinedCmdBuf(null),
            kind: Ctx.CmdBufKind,
        ) void,

        descriptorPoolDestroy: fn (
            self: *Ctx,
            pool: Ctx.DescPool,
        ) void,
        descriptorPoolCreate: fn (
            self: *Ctx,
            comptime max_cmds: u32,
            options: Ctx.DescPool.InitOptions,
        ) Ctx.DescPool,
        descriptorSetsUpdate: fn (
            self: *Ctx,
            comptime max_updates: u32,
            updates: []const Ctx.DescUpdateCmd,
        ) void,

        frameStart: fn (self: *Ctx) void,

        getDevice: fn (self: *const Self) Ctx.Device,

        imageCreate: fn (
            self: *Ctx,
            options: Ctx.ImageOptions,
        ) Ctx.Image(.{}),
        imageDestroy: fn (
            self: *Ctx,
            image: Ctx.Image(.{}),
        ) void,
        imageMemReqs: fn (
            self: *Ctx,
            image: Ctx.Image(.{}),
        ) Ctx.MemReqs,
        imageViewCreate: fn (
            self: *Ctx,
            options: Ctx.ImageView.InitOptions,
        ) Ctx.ImageView,
        imageViewDestroy: fn (
            self: *Ctx,
            image_view: Ctx.ImageView,
        ) void,

        memoryCreate: fn (
            self: *Ctx,
            options: Ctx.MemoryCreateUntypedOptions,
        ) Ctx.Memory(.{}),
        deviceMemoryDestroy: fn (
            self: *Ctx,
            memory: Ctx.Memory(.{}),
        ) void,

        combinedPipelineDestroy: fn (
            self: *Ctx,
            combined: Ctx.CombinedPipeline(null),
        ) void,
        combinedPipelinesCreate: fn (
            self: *Ctx,
            comptime max_cmds: u32,
            cmds: []const Ctx.InitCombinedPipelineCmd,
        ) void,

        present: fn (self: *Ctx) u64,

        acquireNextImage: fn (self: *Ctx) ?Ctx.ImageView,

        samplerCreate: fn (
            self: *Ctx,
            options: Ctx.Sampler.InitOptions,
        ) Ctx.Sampler,
        samplerDestroy: fn (
            self: *Ctx,
            sampler: Ctx.Sampler,
        ) void,

        timestampCalibration: fn (self: *Ctx) Ctx.TimestampCalibration,

        waitIdle: fn (self: *const Ctx) void,

        pub fn create() @This() {
            var self: @This() = undefined;
            inline for (std.meta.fields(@This())) |field| {
                const Expected = field.type;
                const Found = @TypeOf(@field(Self, field.name));
                if (Expected != Found) {
                    @compileError(
                        "expected '" ++
                            field.name ++
                            "' to be type '" ++ @typeName(Expected) ++
                            "', found '" ++
                            @typeName(Found) ++
                            "'",
                    );
                }
                @field(self, field.name) = @field(Self, field.name);
            }
            return self;
        }
    };
}
