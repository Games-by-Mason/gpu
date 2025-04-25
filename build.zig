const builtin = @import("builtin");
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const no_llvm = b.option(
        bool,
        "no-llvm",
        "Don't use the LLVM backend.",
    ) orelse false;

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
        .use_llvm = !no_llvm,
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
        .use_llvm = !no_llvm,
    });
    const docs = docs_exe.getEmittedDocs();
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs,
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    const docs_step = b.step("docs", "Build the docs");
    docs_step.dependOn(&install_docs.step);
}
