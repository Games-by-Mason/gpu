const std = @import("std");
const tracy = @import("tracy");
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
ShaderModule: type,
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
    comptime max_descs: u32,
    options: Ctx.CombinedPipelineLayout.InitOptions,
) Ctx.CombinedPipelineLayout,
combinedPipelineLayoutDestroy: fn (
    self: *Ctx,
    layout: Ctx.CombinedPipelineLayout,
) void,

cmdBufBeginZone: fn (
    self: *Ctx,
    cb: Ctx.CmdBuf,
    loc: *const tracy.SourceLocation,
) void,
cmdBufEndZone: fn (
    self: *Ctx,
    cb: Ctx.CmdBuf,
) void,

imageTransitionUndefinedToTransferDst: fn (
    options: Ctx.ImageTransition.UndefinedToTransferDstOptions,
    out_transition: anytype,
) void,
imageTransitionUndefinedToColorOutputAttachment: fn (
    options: Ctx.ImageTransition.UndefinedToColorOutputAttachmentOptions,
    out_transition: anytype,
) void,
imageTransitionUndefinedToColorOutputAttachmentAfterRead: fn (
    options: Ctx.ImageTransition.UndefinedToColorOutputAttachmentOptionsAfterRead,
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
    cb: Ctx.CmdBuf,
    options: Ctx.CmdBuf.DrawOptions,
) void,
cmdBufTransitionImages: fn (
    self: *Ctx,
    cb: Ctx.CmdBuf,
    transitions: anytype,
) void,
cmdBufUploadImage: fn (
    self: *Ctx,
    cb: Ctx.CmdBuf,
    dst: Ctx.Image(null),
    src: Ctx.Buf(.{}),
    regions_untyped: anytype,
) void,
cmdBufUploadBuffer: fn (
    self: *Ctx,
    cb: Ctx.CmdBuf,
    dst: Ctx.Buf(.{}),
    src: Ctx.Buf(.{}),
    regions_untyped: anytype,
) void,
cmdBufBeginRendering: fn (
    self: *Ctx,
    cb: Ctx.CmdBuf,
    options: Ctx.CmdBuf.BeginRenderingOptions,
) void,
cmdBufEndRendering: fn (
    self: *Ctx,
    cb: Ctx.CmdBuf,
) void,
cmdBufSetViewport: fn (
    self: *Ctx,
    cb: Ctx.CmdBuf,
    viewport: Ctx.Viewport,
) void,
cmdBufSetScissor: fn (
    self: *Ctx,
    cb: Ctx.CmdBuf,
    scissor: Ctx.Rect2D,
) void,
cmdBufBindPipeline: fn (
    self: *Ctx,
    cb: Ctx.CmdBuf,
    combined: Ctx.Pipeline,
) void,
cmdBufBindDescSet: fn (
    self: *Ctx,
    cb: Ctx.CmdBuf,
    pipeline: Ctx.Pipeline,
    set: Ctx.DescSet,
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
    loc: *const tracy.SourceLocation,
) Ctx.CmdBuf,
cmdBufSubmit: fn (
    self: *Ctx,
    combined_command_buffer: Ctx.CmdBuf,
) void,

descPoolDestroy: fn (
    self: *Ctx,
    pool: Ctx.DescPool,
) void,
descPoolCreate: fn (
    self: *Ctx,
    options: Ctx.DescPool.InitOptions,
) Ctx.DescPool,
descSetsUpdate: fn (
    self: *Ctx,
    updates: []const Ctx.DescUpdateCmd,
) void,

beginFrame: fn (self: *Ctx) void,
endFrame: fn (self: *Ctx, options: Ctx.EndFrameOptions) void,
acquireNextImage: fn (
    self: *Ctx,
    framebuf_extent: Ctx.Extent2D,
) Ctx.Attachment,

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

shaderModuleCreate: fn (
    self: *Ctx,
    options: Ctx.ShaderModule.InitOptions,
) Ctx.ShaderModule,
shaderModuleDestroy: fn (
    self: *Ctx,
    module: Ctx.ShaderModule,
) void,

pipelineDestroy: fn (
    self: *Ctx,
    combined: Ctx.Pipeline,
) void,
pipelinesCreate: fn (
    self: *Ctx,
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
