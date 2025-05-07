const std = @import("std");
const Ctx = @import("Ctx.zig");
const Allocator = std.mem.Allocator;

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
ImageTransition: type,
ImageUploadRegion: type,
BufferUploadRegion: type,
InitOptions: type,

init: fn (self: *Ctx, options: anytype) void,
deinit: fn (self: *Ctx, gpa: Allocator) void,

dedicatedBufCreate: fn (
    self: *Ctx,
    name: Ctx.DebugName,
    kind: Ctx.BufKind,
    size: u64,
) Ctx.DedicatedAllocation(Ctx.DedicatedBuf(.{})),
dedicatedUploadBufCreate: fn (
    self: *Ctx,
    name: Ctx.DebugName,
    kind: Ctx.BufKind,
    size: u64,
    prefer_device_local: bool,
) Ctx.DedicatedAllocation(Ctx.DedicatedUploadBuf(.{})),
dedicatedReadbackBufCreate: fn (
    self: *Ctx,
    name: Ctx.DebugName,
    kind: Ctx.BufKind,
    size: u64,
) Ctx.DedicatedAllocation(Ctx.DedicatedReadbackBuf(.{})),

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

imageTransitionUndefinedToTransferDst: fn (
    options: Ctx.ImageTransition.UndefinedToTransferDstOptions,
    out_transition: anytype,
) void,
imageTransitionTransferDstToReadOnly: fn (
    options: Ctx.ImageTransition.TransferDstToReadOnlyOptions,
    out_transition: anytype,
) void,
imageTransitionTransferDstToColorOutputAttachment: fn (
    options: Ctx.ImageTransition.TransferDstToColorOutputAttachmentOptions,
    out_transition: anytype,
) void,
imageTransitionReadOnlyToColorOutputAttachment: fn (
    options: Ctx.ImageTransition.ReadOnlyToColorOutputAttachmentOptions,
    out_transition: anytype,
) void,
imageTransitionColorOutputAttachmentToReadOnly: fn (
    options: Ctx.ImageTransition.ColorOutputAttachmentToReadOnlyOptions,
    out_transition: anytype,
) void,

cmdBufDraw: fn (
    self: *Ctx,
    cb: Ctx.CombinedCmdBuf,
    options: []const Ctx.DrawCmd,
) void,
cmdBufTransitionImages: fn (
    self: *Ctx,
    cb: Ctx.CombinedCmdBuf,
    transitions: anytype,
) void,
cmdBufUploadImage: fn (
    self: *Ctx,
    cb: Ctx.CombinedCmdBuf,
    dst: Ctx.Image(null),
    src: Ctx.Buf(.{}),
    regions_untyped: anytype,
) void,
cmdBufUploadBuffer: fn (
    self: *Ctx,
    cb: Ctx.CombinedCmdBuf,
    dst: Ctx.Buf(.{}),
    src: Ctx.Buf(.{}),
    regions_untyped: anytype,
) void,
cmdBufBeginRendering: fn (
    self: *Ctx,
    cb: Ctx.CmdBuf,
    options: Ctx.CombinedCmdBuf.BeginRenderingOptions,
) void,
cmdBufEndRendering: fn (
    self: *Ctx,
    cb: Ctx.CmdBuf,
) void,

imageUploadRegionInit: fn (
    options: Ctx.ImageUpload.Region.InitOptions,
    out_region: anytype,
) void,
bufferUploadRegionInit: fn (
    options: Ctx.BufferUpload.Region.InitOptions,
    out_region: anytype,
) void,

cmdBufCreate: fn (
    self: *Ctx,
    options: Ctx.CombinedCmdBuf.InitOptions,
) Ctx.CmdBuf,
combinedCmdBufSubmit: fn (
    self: *Ctx,
    combined_command_buffer: Ctx.CombinedCmdBuf,
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

beginFrame: fn (self: *Ctx) void,
endFrame: fn (self: *Ctx, options: Ctx.EndFrameOptions) void,
acquireNextImage: fn (self: *Ctx) Ctx.ImageView,

getDevice: fn (self: *const Ctx) Ctx.Device,

imageCreate: fn (
    self: *Ctx,
    alloc_options: Ctx.Image(null).AllocOptions,
    image_options: Ctx.ImageOptions,
) Ctx.ImageResultUntyped,
imageDestroy: fn (
    self: *Ctx,
    image: Ctx.Image(null),
) void,
imageMemoryRequirements: fn (
    self: *Ctx,
    options: Ctx.ImageOptions,
) Ctx.MemoryRequirements,
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
) Ctx.MemoryUnsized,
memoryDestroy: fn (
    self: *Ctx,
    memory: Ctx.MemoryUnsized,
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
