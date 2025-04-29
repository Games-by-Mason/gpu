const builtin = @import("builtin");
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const native_target = b.resolveTargetQuery(.{});

    const vk_backend = b.addModule("VkBackend", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const gpu = b.dependency("gpu", .{
        .target = target,
        .optimize = optimize,
    });
    vk_backend.addImport("gpu", gpu.module("gpu"));

    const vulkan_headers = b.dependency("vulkan_headers", .{});
    const vulkan_zig = b.dependency("vulkan_zig", .{
        .target = native_target,
        .optimize = optimize,
    });
    const generator = vulkan_zig.artifact("vulkan-zig-generator");
    var run_generator = b.addRunArtifact(generator);
    run_generator.addFileArg(vulkan_headers.path("registry/vk.xml"));
    const vk_zig = run_generator.addOutputFileArg("vk.zig");
    const vulkan = b.addModule("vulkan", .{ .root_source_file = vk_zig });
    vk_backend.addImport("vulkan", vulkan);
}
