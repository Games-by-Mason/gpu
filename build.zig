const builtin = @import("builtin");
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const gpu = b.addModule("gpu", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const tracy = b.dependency("tracy", .{
        .target = target,
        .optimize = optimize,
    });
    gpu.addImport("tracy", tracy.module("tracy"));

    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);

    const check_step = b.step("check", "Check the build");
    check_step.dependOn(&lib_unit_tests.step);

    // We need an executable to generate docs, but we don't want to use a test executable because
    // "test" ends up in our URLs if we do.
    const docs_exe = b.addExecutable(.{
        .name = "gpu",
        .root_source_file = b.path("src/docs.zig"),
        .target = target,
        .optimize = optimize,
    });
    const docs = docs_exe.getEmittedDocs();
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs,
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    const docs_step = b.step("docs", "Build the docs");
    docs_step.dependOn(&install_docs.step);

    buildVulkanBackend(b, target, optimize, gpu);
}

fn buildVulkanBackend(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    gpu: *std.Build.Module,
) void {
    const native_target = b.resolveTargetQuery(.{});

    const vk_backend = b.addModule("VkBackend", .{
        .root_source_file = b.path("backends/vk/src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Needed for `setenv`
    if (target.result.os.tag != .windows) {
        vk_backend.link_libc = true;
    }

    vk_backend.addImport("gpu", gpu);

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
