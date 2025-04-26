pub const pools = @import("pools");

const std = @import("std");
const math = std.math;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const vk = @import("vulkan");
const log = std.log.scoped(.gpu_vk);
const gpu = @import("gpu");
const Ctx = gpu.Ctx;
const tracy = gpu.tracy;
const Zone = tracy.Zone;
const TracyQueue = tracy.GpuQueue;
const global_options = gpu.options;

pub const vulkan = @import("vulkan");

// Context
surface: vk.SurfaceKHR,
base_wrapper: vk.BaseWrapper,
device: vk.DeviceProxy,
instance: vk.InstanceProxy,
physical_device: PhysicalDevice,
swapchain: Swapchain,
debug_messenger: vk.DebugUtilsMessengerEXT,

// Queues
timestamp_period: f32,
combined_queues: Ctx.Device.CombinedQueues,
graphics_queue_family_index: u32,
compute_queue_family_index: ?u32,
transfer_queue_family_index: ?u32,

// Command buffers and render passes
render_pass: vk.RenderPass,
graphics_command_pools: [global_options.max_frames_in_flight]vk.CommandPool,
compute_command_pools: [global_options.max_frames_in_flight]vk.CommandPool,
transfer_command_pools: [global_options.max_frames_in_flight]vk.CommandPool,

// Synchronization
image_availables: [global_options.max_frames_in_flight]vk.Semaphore,
ready_for_present: [global_options.max_frames_in_flight]std.BoundedArray(vk.Semaphore, global_options.max_cmdbufs_per_frame),

// Frame info
image_index: ?u32 = null,

// Tracy info
zones: pools.ArrayBacked(u8, TracyQueue, undefined),
tracy_query_pool: vk.QueryPool,

timestamp_queries: bool,

pub const InitOptions = struct {
    pub const GetInstanceProcAddress = *const fn (
        instance: vk.Instance,
        name: [*:0]const u8,
    ) ?*const fn () callconv(.C) void;
    const CreateSurfaceError = error{
        OutOfHostMemory,
        OutOfDeviceMemory,
        NativeWindowInUseKHR,
        Unknown,
    };

    instance_extensions: [][*:0]const u8,
    getInstanceProcAddress: GetInstanceProcAddress,
    surface_context: ?*anyopaque,
    createSurface: *const fn (
        instance: vk.Instance,
        context: ?*anyopaque,
        allocation_callbacks: ?*const vk.AllocationCallbacks,
    ) CreateSurfaceError!vk.SurfaceKHR,
};

pub const Buf = vk.Buffer;
pub const CmdBuf = vk.CommandBuffer;
pub const DescPool = vk.DescriptorPool;
pub const DescSet = vk.DescriptorSet;
pub const DescSetLayout = vk.DescriptorSetLayout;
pub const Memory = vk.DeviceMemory;
pub const Image = vk.Image;
pub const ImageView = vk.ImageView;
pub const Queue = vk.Queue;
pub const Pipeline = vk.Pipeline;
pub const PipelineLayout = vk.PipelineLayout;
pub const Sampler = vk.Sampler;
pub const Semaphore = vk.Semaphore;

pub fn init(options: Ctx.InitOptionsImpl(InitOptions)) @This() {
    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    log.info("Initializing Vulkan backend", .{});

    const gpa = options.gpa;

    // Load the base dispatch function pointers
    const fp_zone = tracy.Zone.begin(.{ .name = "load fps", .src = @src() });
    const getInstProcAddr = options.backend.getInstanceProcAddress;
    const base_wrapper = vk.BaseWrapper.load(getInstProcAddr);
    fp_zone.end();

    // Determine the required layers and extensions
    const ext_zone = tracy.Zone.begin(.{ .name = "layers & extensions", .src = @src() });
    var layers: std.ArrayListUnmanaged([*:0]const u8) = .{};
    defer layers.deinit(gpa);

    var instance_extensions: std.ArrayListUnmanaged([*:0]const u8) = .{};
    defer instance_extensions.deinit(gpa);
    instance_extensions.appendSlice(gpa, options.backend.instance_extensions) catch @panic("OOM");

    if (options.validation) {
        layers.append(gpa, "VK_LAYER_KHRONOS_validation") catch @panic("OOM");
        instance_extensions.append(gpa, vk.extensions.ext_debug_utils.name) catch @panic("OOM");
    }

    log.info("requested instance extensions: {s}", .{instance_extensions.items});
    log.info("requested layers: {s}", .{layers.items});
    ext_zone.end();

    const inst_handle_zone = tracy.Zone.begin(.{ .name = "create instance handle", .src = @src() });
    const instance_handle = base_wrapper.createInstance(&.{
        .p_application_info = &.{
            .api_version = @bitCast(vk.makeApiVersion(0, 1, 3, 0)),
            .p_application_name = if (options.application_name) |n| n.ptr else null,
            .application_version = @bitCast(vk.makeApiVersion(
                0,
                options.application_version.major,
                options.application_version.minor,
                options.application_version.patch,
            )),
            .p_engine_name = if (options.engine_name) |n| n.ptr else null,
            .engine_version = @bitCast(vk.makeApiVersion(
                0,
                options.engine_version.major,
                options.engine_version.minor,
                options.engine_version.patch,
            )),
        },
        .enabled_layer_count = math.cast(u32, layers.items.len) orelse @panic("overflow"),
        .pp_enabled_layer_names = layers.items.ptr,
        .enabled_extension_count = math.cast(u32, instance_extensions.items.len) orelse @panic("overflow"),
        .pp_enabled_extension_names = instance_extensions.items.ptr,
        .p_next = if (options.validation) &validation_features else null,
    }, null) catch |err| @panic(@errorName(err));
    inst_handle_zone.end();

    const instance_wrapper_zone = tracy.Zone.begin(.{ .name = "create instance wrapper", .src = @src() });
    const instance_wrapper = gpa.create(vk.InstanceWrapper) catch @panic("OOM");
    instance_wrapper.* = vk.InstanceWrapper.load(
        instance_handle,
        base_wrapper.dispatch.vkGetInstanceProcAddr.?,
    );
    const instance_proxy = vk.InstanceProxy.init(instance_handle, instance_wrapper);
    instance_wrapper_zone.end();

    const debug_messenger_zone = tracy.Zone.begin(.{ .name = "create debug messenger", .src = @src() });
    const debug_messenger = if (options.validation) instance_proxy.createDebugUtilsMessengerEXT(
        &create_debug_messenger_info,
        null,
    ) catch |err| @panic(@errorName(err)) else .null_handle;
    debug_messenger_zone.end();

    const surface_zone = tracy.Zone.begin(.{ .name = "create surface", .src = @src() });
    const surface = options.backend.createSurface(
        instance_proxy.handle,
        options.backend.surface_context,
        null,
    ) catch |err| @panic(@errorName(err));
    surface_zone.end();

    const devices_zone = tracy.Zone.begin(.{ .name = "pick device", .src = @src() });
    const enumerate_devices_zone = tracy.Zone.begin(.{ .name = "enumerate", .src = @src() });
    const physical_devices = instance_proxy.enumeratePhysicalDevicesAlloc(gpa) catch |err| @panic(@errorName(err));
    enumerate_devices_zone.end();
    defer gpa.free(physical_devices);

    var best_physical_device: PhysicalDevice = .{};
    var required_device_extensions: std.ArrayListUnmanaged([*:0]const u8) = .{};
    defer required_device_extensions.deinit(gpa);
    required_device_extensions.append(gpa, vk.extensions.khr_swapchain.name) catch @panic("OOM");
    if (options.timestamp_queries) {
        required_device_extensions.append(gpa, vk.extensions.khr_calibrated_timestamps.name) catch @panic("OOM");
    }

    log.info("enumerate devices:", .{});
    for (physical_devices, 0..) |device, i| {
        const properties = instance_proxy.getPhysicalDeviceProperties(device);

        var host_query_reset_features: vk.PhysicalDeviceHostQueryResetFeatures = .{};
        var shader_draw_parameters_features: vk.PhysicalDeviceShaderDrawParametersFeatures = .{
            .p_next = &host_query_reset_features,
        };
        var features: vk.PhysicalDeviceFeatures2 = .{
            .features = .{},
            .p_next = &shader_draw_parameters_features,
        };
        instance_proxy.getPhysicalDeviceFeatures2(device, &features);
        const supports_required_features =
            // Roadmap 2022
            shader_draw_parameters_features.shader_draw_parameters == vk.TRUE and
            // Only required when using device timers
            (!options.timestamp_queries or host_query_reset_features.host_query_reset == vk.TRUE) and
            // Roadmap 2024
            features.features.multi_draw_indirect == vk.TRUE and
            // Roadmap 2022
            features.features.draw_indirect_first_instance == vk.TRUE;
        const all_features = .{
            shader_draw_parameters_features,
            host_query_reset_features,
            features,
        };

        const rank: u8 = options.device_type_ranks.get(switch (properties.device_type) {
            .discrete_gpu => .discrete,
            .integrated_gpu => .integrated,
            .virtual_gpu => .virtual,
            .cpu => .cpu,
            else => .other,
        });

        const queue_family_properties = instance_proxy.getPhysicalDeviceQueueFamilyPropertiesAlloc(device, gpa) catch |err| @panic(@errorName(err));
        defer gpa.free(queue_family_properties);
        const supports_graphics_present = for (queue_family_properties, 0..) |qfp, qfi| {
            if (queueFamilyHasGraphics(qfp) and queueFamilyHasPresent(instance_proxy, device, surface, @intCast(qfi))) {
                break true;
            }
        } else false;

        var missing_device_extensions: std.StringHashMapUnmanaged(void) = .{};
        defer missing_device_extensions.deinit(gpa);
        for (required_device_extensions.items) |required| missing_device_extensions.put(gpa, std.mem.span(required), {}) catch |err| @panic(@errorName(err));

        const supported_device_extensions = instance_proxy.enumerateDeviceExtensionPropertiesAlloc(device, null, gpa) catch |err| @panic(@errorName(err));
        defer gpa.free(supported_device_extensions);
        for (supported_device_extensions) |extension_properties| {
            const name: [*c]const u8 = @ptrCast(extension_properties.extension_name[0..]);
            _ = missing_device_extensions.remove(std.mem.span(name));
        }
        const extensions_supported = missing_device_extensions.count() == 0;

        const surface_capabilities, const surface_format, const present_mode = if (extensions_supported) b: {
            const surface_capabilities = instance_proxy.getPhysicalDeviceSurfaceCapabilitiesKHR(
                device,
                surface,
            ) catch |err| @panic(@errorName(err));

            var best_surface_format: ?vk.SurfaceFormatKHR = null;
            var best_surface_format_rank: u8 = 0;
            const surface_formats = instance_proxy.getPhysicalDeviceSurfaceFormatsAllocKHR(
                device,
                surface,
                gpa,
            ) catch |err| @panic(@errorName(err));
            defer gpa.free(surface_formats);
            for (surface_formats) |surface_format| {
                var format_rank: u8 = 0;
                switch (surface_format.format) {
                    .b8g8r8a8_srgb => format_rank += 3,
                    .r8g8b8a8_srgb => format_rank += 2,
                    else => {},
                }
                switch (surface_format.color_space) {
                    .srgb_nonlinear_khr => format_rank += 3,
                    else => {},
                }

                if (best_surface_format == null or format_rank > best_surface_format_rank) {
                    best_surface_format = surface_format;
                    best_surface_format_rank = format_rank;
                }
            }
            var best_present_mode: vk.PresentModeKHR = .fifo_khr;
            const present_modes = instance_proxy.getPhysicalDeviceSurfacePresentModesAllocKHR(
                device,
                surface,
                gpa,
            ) catch |err| @panic(@errorName(err));
            defer gpa.free(present_modes);
            for (present_modes) |present_mode| {
                switch (present_mode) {
                    .fifo_relaxed_khr => best_present_mode = present_mode,
                    else => {},
                }
            }
            break :b .{ surface_capabilities, best_surface_format, best_present_mode };
        } else .{ null, null, null };

        log.info("  {}. {s}:", .{ i, bufToStr(&properties.device_name) });
        log.info("    * device type: {}", .{properties.device_type});
        log.info("    * has graphics+present queue: {?}", .{supports_graphics_present});
        log.info("    * present mode: {?}", .{present_mode});
        log.info("    * surface format: {?}", .{surface_format});
        log.info("    * features: {}", .{all_features});
        log.info("    * max indirect draw count: {}", .{properties.limits.max_draw_indirect_count});
        log.info("    * min uniform buffer offset alignment: {}", .{properties.limits.min_uniform_buffer_offset_alignment});
        log.info("    * min storage buffer offset alignment: {}", .{properties.limits.min_storage_buffer_offset_alignment});
        if (!extensions_supported) {
            log.info("    * missing extensions:", .{});
            var key_iterator = missing_device_extensions.keyIterator();
            while (key_iterator.next()) |key| {
                log.info("      * {s}", .{key.*});
            }
        }

        const composite_alpha: ?vk.CompositeAlphaFlagsKHR = b: {
            if (surface_capabilities) |sc| {
                const supported = sc.supported_composite_alpha;
                log.info("    * supported composite alpha: {}", .{supported});
                if (supported.opaque_bit_khr) {
                    break :b .{ .opaque_bit_khr = true };
                } else if (supported.inherit_bit_khr) {
                    break :b .{ .inherit_bit_khr = true };
                } else if (supported.pre_multiplied_bit_khr) {
                    break :b .{ .pre_multiplied_bit_khr = true };
                } else if (supported.post_multiplied_bit_khr) {
                    break :b .{ .post_multiplied_bit_khr = true };
                } else {
                    break :b null;
                }
            }
            break :b null;
        };

        const compatible = supports_graphics_present and extensions_supported and composite_alpha != null and supports_required_features;

        if (compatible) {
            log.info("    * rank: {}", .{rank});
        } else {
            log.info("    * rank: incompatible", .{});
        }

        if (compatible and rank > best_physical_device.rank) {
            best_physical_device = .{
                .device = device,
                .name = properties.device_name,
                .index = i,
                .rank = rank,
                .surface_format = surface_format orelse continue,
                .present_mode = present_mode orelse continue,
                .surface_capabilities = surface_capabilities orelse continue,
                .composite_alpha = composite_alpha.?,
                .ty = switch (properties.device_type) {
                    .integrated_gpu => .integrated,
                    .discrete_gpu => .discrete,
                    .virtual_gpu => .virtual,
                    .cpu => .cpu,
                    .other => .other,
                    _ => .other,
                },
                // Cast safe because highest either can be is 256
                // https://registry.khronos.org/vulkan/specs/1.3/html/chap33.html#limits-minmax
                .min_uniform_buffer_offset_alignment = @intCast(properties.limits.min_uniform_buffer_offset_alignment),
                .min_storage_buffer_offset_alignment = @intCast(properties.limits.min_storage_buffer_offset_alignment),
                .max_draw_indirect_count = properties.limits.max_draw_indirect_count,
                .sampler_anisotropy = features.features.sampler_anisotropy == vk.TRUE,
                .max_sampler_anisotropy = properties.limits.max_sampler_anisotropy,
            };
        }
    }

    if (best_physical_device.device == .null_handle) {
        @panic("NoSupportedDevices");
    }

    log.info("device {} chosen: {s}", .{ best_physical_device.index, bufToStr(&best_physical_device.name) });
    devices_zone.end();

    // Iterate over the available queues, and find indices for the various queue types requested
    const queue_zone = tracy.Zone.begin(.{ .name = "queue setup", .src = @src() });
    const queue_family_properties = instance_proxy.getPhysicalDeviceQueueFamilyPropertiesAlloc(best_physical_device.device, gpa) catch |err| @panic(@errorName(err));
    defer gpa.free(queue_family_properties);

    const graphics_queue_family_index: u32 = for (queue_family_properties, 0..) |qfp, qfi| {
        if (queueFamilyHasGraphics(qfp) and
            queueFamilyHasPresent(instance_proxy, best_physical_device.device, surface, @intCast(qfi)) and
            queueFamilyHasCompute(qfp) and
            queueFamilyHasTransfer(qfp))
        {
            break @intCast(qfi);
        }
    } else {
        // If none existed, we would have ruled out this device
        unreachable;
    };

    const compute_queue_family_index: ?u32 = for (queue_family_properties, 0..) |qfp, qfi| {
        if (queueFamilyHasCompute(qfp) and
            queueFamilyHasTransfer(qfp) and
            !queueFamilyHasGraphics(qfp))
        {
            break @intCast(qfi);
        }
    } else null;

    const transfer_queue_family_index: ?u32 = for (queue_family_properties, 0..) |qfp, qfi| {
        if (queueFamilyHasTransfer(qfp) and
            !queueFamilyHasGraphics(qfp) and
            !queueFamilyHasCompute(qfp))
        {
            break @intCast(qfi);
        }
    } else null;

    const queue_family_allocated: []u8 = gpa.alloc(u8, queue_family_properties.len) catch @panic("OOM");
    defer gpa.free(queue_family_allocated);
    for (queue_family_allocated) |*c| c.* = 0;

    const queue_priorities: [gpu.max_queues]f32 = .{1.0} ** gpu.max_queues;
    var queue_create_infos: std.BoundedArray(vk.DeviceQueueCreateInfo, 3) = .{};
    {
        const queue_count = @min(
            queue_family_properties[graphics_queue_family_index].queue_count,
            global_options.max_queues.graphics,
        );
        log.info("graphics queues: {} queue(s) from queue family {}", .{ queue_count, graphics_queue_family_index });
        queue_create_infos.appendAssumeCapacity(.{
            .queue_family_index = graphics_queue_family_index,
            .queue_count = queue_count,
            .p_queue_priorities = queue_priorities[0..].ptr,
        });
    }
    if (compute_queue_family_index) |qfi| {
        const queue_count = @min(
            queue_family_properties[qfi].queue_count,
            global_options.max_queues.compute,
        );
        log.info("compute queues: {} queue(s) from queue family {}", .{ queue_count, qfi });
        queue_create_infos.appendAssumeCapacity(.{
            .queue_family_index = qfi,
            .queue_count = queue_count,
            .p_queue_priorities = queue_priorities[0..].ptr,
        });
    }
    if (transfer_queue_family_index) |qfi| {
        const queue_count = @min(
            queue_family_properties[qfi].queue_count,
            global_options.max_queues.transfer,
        );
        log.info("transfer queues: {} queue(s) from queue family {}", .{ queue_count, qfi });
        queue_create_infos.appendAssumeCapacity(.{
            .queue_family_index = qfi,
            .queue_count = queue_count,
            .p_queue_priorities = queue_priorities[0..].ptr,
        });
    }
    queue_zone.end();

    const device_proxy_zone = tracy.Zone.begin(.{ .name = "create device proxy", .src = @src() });
    var device_features_11: vk.PhysicalDeviceVulkan11Features = .{
        .shader_draw_parameters = vk.TRUE,
    };
    const device_features_12: vk.PhysicalDeviceVulkan12Features = .{
        .host_query_reset = @intFromBool(options.timestamp_queries),
        // Party of DX12, so it seems reasonable to assume all Vulkan 1.3 hardware supports these in
        // practice.
        .timeline_semaphore = vk.TRUE,
        .p_next = &device_features_11,
    };
    const device_features: vk.PhysicalDeviceFeatures = .{
        .draw_indirect_first_instance = vk.TRUE,
        .multi_draw_indirect = vk.TRUE,
        .sampler_anisotropy = @intFromBool(best_physical_device.sampler_anisotropy),
    };
    const device_create_info: vk.DeviceCreateInfo = .{
        .p_queue_create_infos = &queue_create_infos.buffer,
        .queue_create_info_count = @intCast(queue_create_infos.len),
        .p_enabled_features = &device_features,
        .enabled_extension_count = @intCast(required_device_extensions.items.len),
        .pp_enabled_extension_names = required_device_extensions.items.ptr,
        .p_next = &device_features_12,
    };
    const device_handle = instance_proxy.createDevice(
        best_physical_device.device,
        &device_create_info,
        null,
    ) catch |err| @panic(@errorName(err));
    device_proxy_zone.end();

    const device_wrapper_zone = tracy.Zone.begin(.{ .name = "create device wrapper", .src = @src() });
    const device_wrapper = gpa.create(vk.DeviceWrapper) catch @panic("OOM");
    device_wrapper.* = vk.DeviceWrapper.load(
        device_handle,
        instance_proxy.wrapper.dispatch.vkGetDeviceProcAddr.?,
    );
    const device = vk.DeviceProxy.init(device_handle, device_wrapper);
    device_wrapper_zone.end();

    const timestamp_zone = tracy.Zone.begin(.{ .name = "gpu timestamp setup", .src = @src() });
    const timestamp_period = if (options.timestamp_queries) b: {
        const properties = instance_proxy.getPhysicalDeviceProperties(best_physical_device.device);

        if (options.timestamp_queries and properties.limits.timestamp_period <= 0) {
            log.err("timestamp queries requested but not supported", .{});
            break :b 0.0;
        }

        if (options.timestamp_queries and properties.limits.timestamp_compute_and_graphics != vk.TRUE) {
            log.err("timestamp queries not supported on compute and graphics (can be relaxed by checking as needed)", .{});
            break :b 0.0;
        }

        break :b properties.limits.timestamp_period;
    } else 0.0;

    const calibration = timestampCalibrationImpl(device, options.timestamp_queries);
    var tracy_queues: u8 = 0;
    timestamp_zone.end();

    var combined_queues: Ctx.Device.CombinedQueues = .{
        .graphics_family = .{},
        .compute_family = .{},
        .transfer_family = .{},
    };
    const get_queue_zone = tracy.Zone.begin(.{ .name = "get queues", .src = @src() });
    {
        const qfi = graphics_queue_family_index;
        inline for (0..global_options.max_queues.graphics) |q| {
            if (q >= queue_family_properties[qfi].queue_count) break;

            const queue = device.getDeviceQueue(
                qfi,
                @intCast(q),
            );
            setName(device, queue, .{ .str = "Graphics Queue", .index = q }, options.validation);
            combined_queues.graphics_family.appendAssumeCapacity(.{
                .queue = .fromBackendType(queue),
                .tracy_queue = TracyQueue.init(.{
                    .gpu_time = calibration.gpu,
                    .period = timestamp_period,
                    .context = tracy_queues,
                    .flags = .{},
                    .type = .vulkan,
                    .name = std.fmt.comptimePrint("Graphics Queue {}", .{q}),
                }),
            });
            tracy_queues += 1;
        }
    }
    if (compute_queue_family_index) |qfi| {
        inline for (0..global_options.max_queues.compute) |q| {
            if (q >= queue_family_properties[qfi].queue_count) break;
            const queue = device.getDeviceQueue(
                qfi,
                @intCast(q),
            );
            setName(device, queue, .{ .str = "Compute Queue", .index = q }, options.validation);
            combined_queues.compute_family.appendAssumeCapacity(.{
                .queue = .fromBackendType(queue),
                .tracy_queue = TracyQueue.init(.{
                    .gpu_time = calibration.gpu,
                    .period = timestamp_period,
                    .context = tracy_queues,
                    .flags = .{},
                    .type = .vulkan,
                    .name = std.fmt.comptimePrint("Compute Queue {}", .{q}),
                }),
            });
            tracy_queues += 1;
        }
    }
    if (transfer_queue_family_index) |qfi| {
        inline for (0..global_options.max_queues.transfer) |q| {
            if (q >= queue_family_properties[qfi].queue_count) break;
            const queue = device.getDeviceQueue(
                qfi,
                @intCast(q),
            );
            setName(device, queue, .{ .str = "Transfer Queue", .index = q }, options.validation);
            combined_queues.transfer_family.appendAssumeCapacity(.{
                .queue = .fromBackendType(queue),
                .tracy_queue = TracyQueue.init(.{
                    .gpu_time = calibration.gpu,
                    .period = timestamp_period,
                    .context = tracy_queues,
                    .flags = .{},
                    .type = .vulkan,
                    .name = std.fmt.comptimePrint("Transfer Queue {}", .{q}),
                }),
            });
            tracy_queues += 1;
        }
    }
    get_queue_zone.end();

    const render_pass_zone = tracy.Zone.begin(.{ .name = "create render pass", .src = @src() });
    const color_attachments = [_]vk.AttachmentDescription{
        .{
            .format = best_physical_device.surface_format.format,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .store,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .undefined,
            .final_layout = .present_src_khr,
        },
    };

    const color_attachment_refs = [_]vk.AttachmentReference{
        .{
            .attachment = 0,
            .layout = .color_attachment_optimal,
        },
    };

    const subpasses = [_]vk.SubpassDescription{
        .{
            .pipeline_bind_point = .graphics,
            .color_attachment_count = color_attachment_refs.len,
            .p_color_attachments = &color_attachment_refs,
        },
    };

    const subpass_dependencise = [_]vk.SubpassDependency{.{
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{ .color_attachment_output_bit = true },
        .src_access_mask = .{},
        .dst_stage_mask = .{ .color_attachment_output_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true },
    }};

    const render_pass_info: vk.RenderPassCreateInfo = .{
        .attachment_count = color_attachments.len,
        .p_attachments = &color_attachments,
        .subpass_count = subpasses.len,
        .p_subpasses = &subpasses,
        .dependency_count = subpass_dependencise.len,
        .p_dependencies = &subpass_dependencise,
    };

    const render_pass = device.createRenderPass(&render_pass_info, null) catch |err| @panic(@errorName(err));
    setName(device, render_pass, .{ .str = "Main" }, options.validation);
    render_pass_zone.end();

    const swapchain = Swapchain.init(
        instance_proxy,
        options.framebuf_size,
        device,
        best_physical_device,
        surface,
        render_pass,
        .null_handle,
        options.validation,
    );

    const command_pools_zone = tracy.Zone.begin(.{ .name = "create command pools", .src = @src() });
    var graphics_command_pools: [global_options.max_frames_in_flight]vk.CommandPool = undefined;
    for (&graphics_command_pools, 0..) |*pool, i| {
        pool.* = device.createCommandPool(&.{
            .flags = .{ .transient_bit = true },
            .queue_family_index = graphics_queue_family_index,
        }, null) catch |err| @panic(@errorName(err));
        setName(device, pool.*, .{ .str = "Graphics", .index = i }, options.validation);
    }

    var compute_command_pools: [global_options.max_frames_in_flight]vk.CommandPool = undefined;
    for (&compute_command_pools, 0..) |*pool, i| {
        pool.* = device.createCommandPool(&.{
            .flags = .{ .transient_bit = true },
            .queue_family_index = compute_queue_family_index orelse graphics_queue_family_index,
        }, null) catch |err| @panic(@errorName(err));
        setName(device, pool.*, .{ .str = "Compute", .index = i }, options.validation);
    }

    var transfer_command_pools: [global_options.max_frames_in_flight]vk.CommandPool = undefined;
    for (&transfer_command_pools, 0..) |*pool, i| {
        pool.* = device.createCommandPool(&.{
            .flags = .{ .transient_bit = true },
            .queue_family_index = transfer_queue_family_index orelse compute_queue_family_index orelse graphics_queue_family_index,
        }, null) catch |err| @panic(@errorName(err));
        setName(device, pool.*, .{ .str = "Transfer", .index = i }, options.validation);
    }
    command_pools_zone.end();

    const sync_primitives_zone = tracy.Zone.begin(.{ .name = "create sync primitives", .src = @src() });
    var image_availables: [global_options.max_frames_in_flight]vk.Semaphore = undefined;
    for (0..global_options.max_frames_in_flight) |i| {
        image_availables[i] = device.createSemaphore(&.{}, null) catch |err| @panic(@errorName(err));
        setName(device, image_availables[i], .{ .str = "Image Available", .index = i }, options.validation);
    }

    var ready_for_present: [global_options.max_frames_in_flight]std.BoundedArray(vk.Semaphore, global_options.max_cmdbufs_per_frame) = @splat(.{});
    for (&ready_for_present, 0..) |*semaphores, frame| {
        for (0..global_options.max_cmdbufs_per_frame) |cb| {
            const semaphore = device.createSemaphore(&.{}, null) catch |err| @panic(@errorName(err));
            setName(device, semaphore, .{
                .str = "Ready For Present",
                .index = frame * global_options.max_cmdbufs_per_frame * cb,
            }, options.validation);
            semaphores.appendAssumeCapacity(semaphore);
        }
    }
    sync_primitives_zone.end();

    const tracy_query_pool = if (tracy.enabled and
        global_options.tracy_query_pool_capacity > 0)
    b: {
        const query_pool_zone = tracy.Zone.begin(.{ .name = "create tracy query pool", .src = @src() });
        defer query_pool_zone.end();

        const tracy_query_pool = device.createQueryPool(&.{
            .query_type = .timestamp,
            .query_count = global_options.tracy_query_pool_capacity,
        }, null) catch |err| @panic(@errorName(err));
        setName(device, tracy_query_pool, .{ .str = "Tracy" }, options.validation);
        device.resetQueryPool(
            tracy_query_pool,
            0,
            global_options.tracy_query_pool_capacity,
        );
        break :b tracy_query_pool;
    } else .null_handle;

    const zones_zone = tracy.Zone.begin(.{ .name = "create zones", .src = @src() });
    const zones = pools.ArrayBacked(u8, TracyQueue, undefined).init(gpa) catch |err| @panic(@errorName(err));
    zones_zone.end();

    return .{
        .surface = surface,
        .base_wrapper = base_wrapper,
        .debug_messenger = debug_messenger,
        .instance = instance_proxy,
        .device = device,
        .swapchain = swapchain,
        .render_pass = render_pass,
        .graphics_command_pools = graphics_command_pools,
        .compute_command_pools = compute_command_pools,
        .transfer_command_pools = transfer_command_pools,
        .image_availables = image_availables,
        .ready_for_present = ready_for_present,
        .physical_device = best_physical_device,
        .timestamp_period = timestamp_period,
        .combined_queues = combined_queues,
        .graphics_queue_family_index = graphics_queue_family_index,
        .transfer_queue_family_index = transfer_queue_family_index,
        .compute_queue_family_index = compute_queue_family_index,
        .zones = zones,
        .tracy_query_pool = tracy_query_pool,
        .timestamp_queries = options.timestamp_queries,
    };
}

pub fn deinit(self: *Ctx, gpa: Allocator) void {
    // Destroy the tracy query pool
    self.backend.device.destroyQueryPool(self.backend.tracy_query_pool, null);
    self.backend.zones.deinit(gpa);

    // Destroy internal sync state
    for (self.backend.ready_for_present) |semaphores| {
        for (semaphores.buffer) |semaphore| {
            self.backend.device.destroySemaphore(semaphore, null);
        }
    }
    for (self.backend.image_availables) |s| self.backend.device.destroySemaphore(s, null);

    // Destroy command state
    for (self.backend.graphics_command_pools) |pool| {
        self.backend.device.destroyCommandPool(pool, null);
    }
    for (self.backend.compute_command_pools) |pool| {
        self.backend.device.destroyCommandPool(pool, null);
    }
    for (self.backend.transfer_command_pools) |pool| {
        self.backend.device.destroyCommandPool(pool, null);
    }

    // Destroy render pass state
    self.backend.device.destroyRenderPass(self.backend.render_pass, null);

    // Destroy swapchain state
    self.backend.swapchain.deinit(self.backend.device);

    // Destroy device state
    self.backend.device.destroyDevice(null);
    gpa.destroy(self.backend.device.wrapper);

    // Destroy the surface
    self.backend.instance.destroySurfaceKHR(self.backend.surface, null);

    // Destroy the debug messenger
    if (self.backend.debug_messenger != .null_handle) {
        self.backend.instance.destroyDebugUtilsMessengerEXT(self.backend.debug_messenger, null);
    }

    // Destroy the instance
    self.backend.instance.destroyInstance(null);
    gpa.destroy(self.backend.instance.wrapper);

    // Mark as undefined
    self.backend = undefined;
}

pub fn dedicatedBufCreate(
    self: *Ctx,
    name: Ctx.DebugName,
    kind: Ctx.BufKind,
    size: u64,
) Ctx.DedicatedBuf(.{}) {
    // Create the buffer
    const usage_flags = bufUsageFlagsFromKind(kind);
    const buffer = self.backend.device.createBuffer(&.{
        .size = size,
        .usage = usage_flags,
        .sharing_mode = .exclusive,
        .flags = .{},
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.device, buffer, name, self.backend.debug_messenger != .null_handle);

    // Allocate memory for the buffer
    const reqs = self.backend.device.getBufferMemoryRequirements(buffer);
    const memory_type_bits: std.bit_set.IntegerBitSet(32) = .{
        .mask = reqs.memory_type_bits,
    };
    const device_memory_properties = self.backend.instance.getPhysicalDeviceMemoryProperties(
        self.backend.physical_device.device,
    );
    const memory_type_index = findMemoryType(
        device_memory_properties,
        memory_type_bits,
        .none,
    ) orelse @panic("unsupported memory type");
    const memory = self.backend.device.allocateMemory(&.{
        .allocation_size = reqs.size,
        .memory_type_index = memory_type_index,
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.device, memory, name, self.backend.debug_messenger != .null_handle);

    // Bind the buffer to the memory
    self.backend.device.bindBufferMemory(
        buffer,
        memory,
        0,
    ) catch |err| @panic(@errorName(err));

    // Return the dedicated buffer
    return .{
        .buf = .fromBackendType(buffer),
        .memory = .fromBackendType(memory),
    };
}

pub fn dedicatedUploadBufCreate(
    self: *Ctx,
    name: Ctx.DebugName,
    kind: Ctx.BufKind,
    size: u64,
    prefer_device_local: bool,
) Ctx.DedicatedUploadBuf(.{}) {
    // Create the buffer
    const usage = bufUsageFlagsFromKind(kind);
    const buffer = self.backend.device.createBuffer(&.{
        .size = size,
        .usage = usage,
        .sharing_mode = .exclusive,
        .flags = .{},
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.device, buffer, name, self.backend.debug_messenger != .null_handle);

    // Create the memory
    const reqs = self.backend.device.getBufferMemoryRequirements(buffer);
    const memory_type_bits: std.bit_set.IntegerBitSet(32) = .{
        .mask = reqs.memory_type_bits,
    };
    const device_memory_properties = self.backend.instance.getPhysicalDeviceMemoryProperties(
        self.backend.physical_device.device,
    );
    const memory_type_index = findMemoryType(
        device_memory_properties,
        memory_type_bits,
        .{ .write = .{ .prefer_device_local = prefer_device_local } },
    ) orelse @panic("unsupported memory type");
    const memory = self.backend.device.allocateMemory(&.{
        .allocation_size = reqs.size,
        .memory_type_index = memory_type_index,
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.device, memory, name, self.backend.debug_messenger != .null_handle);

    // Bind the buffer to the memory
    self.backend.device.bindBufferMemory(
        buffer,
        memory,
        0,
    ) catch |err| @panic(@errorName(err));

    // Map the memory
    const mapping: [*]u8 = @ptrCast(self.backend.device.mapMemory(
        memory,
        0,
        size,
        .{},
    ) catch |err| @panic(@errorName(err)).?);

    // Return the dedicated buffer
    var data: []volatile anyopaque = undefined;
    data.ptr = @ptrCast(mapping);
    data.len = size;
    return .{
        .buf = .fromBackendType(buffer),
        .memory = .fromBackendType(memory),
        .data = data,
    };
}

pub fn dedicatedReadbackBufCreate(
    self: *Ctx,
    name: Ctx.DebugName,
    kind: Ctx.BufKind,
    size: u64,
) Ctx.DedicatedReadbackBuf(.{}) {
    // Create the buffer
    const buffer = self.backend.device.createBuffer(&.{
        .size = size,
        .usage = bufUsageFlagsFromKind(kind),
        .sharing_mode = .exclusive,
        .flags = .{},
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.device, buffer, name, self.backend.debug_messenger != .null_handle);

    // Create the memory
    const reqs = self.backend.device.getBufferMemoryRequirements(buffer);
    const memory_type_bits: std.bit_set.IntegerBitSet(32) = .{
        .mask = reqs.memory_type_bits,
    };
    const device_memory_properties = self.backend.instance.getPhysicalDeviceMemoryProperties(
        self.backend.physical_device.device,
    );
    const memory_type_index = findMemoryType(
        device_memory_properties,
        memory_type_bits,
        .read,
    ) orelse @panic("unsupported memory type");
    const memory = self.backend.device.allocateMemory(&.{
        .allocation_size = reqs.size,
        .memory_type_index = memory_type_index,
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.device, memory, name, self.backend.debug_messenger != .null_handle);

    // Bind the buffer to the memory
    self.backend.device.bindBufferMemory(
        buffer,
        memory,
        0,
    ) catch |err| @panic(@errorName(err));

    // Map the buffer
    const mapping: [*]u8 = @ptrCast(self.backend.device.mapMemory(
        memory,
        0,
        size,
        .{},
    ) catch |err| @panic(@errorName(err)).?);

    // Return the dedicated buffer
    return .{
        .buf = .fromBackendType(buffer),
        .memory = .fromBackendType(memory),
        .data = mapping[0..size],
    };
}

fn bufUsageFlagsFromKind(kind: Ctx.BufKind) vk.BufferUsageFlags {
    const result: vk.BufferUsageFlags = .{
        .transfer_src_bit = kind.transfer_src,
        .transfer_dst_bit = kind.transfer_dst,
        .uniform_texel_buffer_bit = kind.uniform_texel,
        .storage_texel_buffer_bit = kind.storage_texel,
        .uniform_buffer_bit = kind.uniform,
        .storage_buffer_bit = kind.storage,
        .index_buffer_bit = kind.index,
        .vertex_buffer_bit = kind.vertex,
        .indirect_buffer_bit = kind.indirect,
        .shader_device_address_bit = kind.shader_device_address,
    };
    assert(@as(u32, @bitCast(result)) != 0);
    return result;
}

pub fn bufDestroy(self: *Ctx, buffer: Ctx.Buf(.{})) void {
    self.backend.device.destroyBuffer(buffer.asBackendType(), null);
}

pub fn combinedPipelineLayoutCreate(
    self: *Ctx,
    comptime max_descriptors: u32,
    options: Ctx.CombinedPipelineLayout.InitOptions,
) Ctx.CombinedPipelineLayout {
    // Create the descriptor set layout
    var descriptors: std.BoundedArray(vk.DescriptorSetLayoutBinding, max_descriptors) = .{};
    for (options.descriptors, 0..) |descriptor, i| {
        descriptors.append(.{
            .binding = @intCast(i),
            .descriptor_type = switch (descriptor.kind) {
                .uniform_buffer => .uniform_buffer,
                .storage_buffer => .storage_buffer,
                .combined_image_sampler => .combined_image_sampler,
            },
            .descriptor_count = descriptor.count,
            .stage_flags = .{
                .vertex_bit = descriptor.stages.vertex,
                .fragment_bit = descriptor.stages.fragment,
            },
            .p_immutable_samplers = null,
        }) catch @panic("OOB");
    }
    const descriptor_set_layout = self.backend.device.createDescriptorSetLayout(&.{
        .binding_count = @intCast(descriptors.len),
        .p_bindings = &descriptors.buffer,
    }, null) catch @panic("OOM");
    setName(self.backend.device, descriptor_set_layout, options.name, self.backend.debug_messenger != .null_handle);

    // Create the pipeline layout
    const pipeline_layout = self.backend.device.createPipelineLayout(&.{
        .set_layout_count = 1,
        .p_set_layouts = &.{descriptor_set_layout},
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.device, pipeline_layout, options.name, self.backend.debug_messenger != .null_handle);

    return .{
        .descriptor_set = .fromBackendType(descriptor_set_layout),
        .pipeline = .fromBackendType(pipeline_layout),
    };
}

pub fn combinedPipelineLayoutDestroy(
    self: *Ctx,
    layout: Ctx.CombinedPipelineLayout,
) void {
    self.backend.device.destroyPipelineLayout(layout.pipeline.asBackendType(), null);
    self.backend.device.destroyDescriptorSetLayout(layout.descriptor_set.asBackendType(), null);
}

pub fn zoneBegin(
    self: *Ctx,
    options: Ctx.Zone.BeginOptions,
) Ctx.Zone {
    if (self.backend.debug_messenger != .null_handle) {
        self.backend.device.cmdBeginDebugUtilsLabelEXT(options.command_buffer.asBackendType(), &.{
            .p_label_name = options.loc.name orelse options.loc.function,
            .color = .{
                @as(f32, @floatFromInt(options.loc.color.r)) / 255.0,
                @as(f32, @floatFromInt(options.loc.color.g)) / 255.0,
                @as(f32, @floatFromInt(options.loc.color.b)) / 255.0,
                @as(f32, @floatFromInt(options.loc.color.a)) / 255.0,
            },
        });
    }
    if (tracy.enabled and self.backend.tracy_query_pool != .null_handle) {
        const zone: Ctx.Zone = .{
            .index = self.backend.zones.put(options.tracy_queue) catch @panic("tracy query pool full"),
        };
        const begin_query = zone.beginId();
        options.tracy_queue.beginZone(.{
            .query_id = begin_query,
            .loc = options.loc,
        });
        self.backend.device.cmdWriteTimestamp(
            options.command_buffer.asBackendType(),
            .{ .top_of_pipe_bit = true },
            self.backend.tracy_query_pool,
            begin_query,
        );
        return zone;
    }
    return .{ .index = 0 };
}

pub fn combinedCmdBufCreate(
    self: *Ctx,
    options: Ctx.CombinedCmdBufCreateOptions,
) Ctx.CombinedCmdBuf(null) {
    var command_buffers = [_]vk.CommandBuffer{.null_handle};
    self.backend.device.allocateCommandBuffers(&.{
        .command_pool = switch (options.kind) {
            .graphics, .present => self.backend.graphics_command_pools[self.frameInFlight()],
            .compute => self.backend.compute_command_pools[self.frameInFlight()],
            .transfer => self.backend.transfer_command_pools[self.frameInFlight()],
        },
        .level = .primary,
        .command_buffer_count = command_buffers.len,
    }, &command_buffers) catch |err| @panic(@errorName(err));
    const command_buffer = command_buffers[0];
    setName(self.backend.device, command_buffer, .{
        .str = options.loc.name orelse options.loc.function,
    }, self.backend.debug_messenger != .null_handle);

    self.backend.device.beginCommandBuffer(command_buffer, &.{
        .flags = .{ .one_time_submit_bit = true },
        .p_inheritance_info = null,
    }) catch |err| @panic(@errorName(err));

    const zone = Ctx.Zone.begin(self, .{
        .command_buffer = .fromBackendType(command_buffer),
        .tracy_queue = options.combined_queue.tracy_queue,
        .loc = options.loc,
    });

    const clear_values = [_]vk.ClearValue{
        .{
            .color = .{ .float_32 = .{ 0.5, 0.5, 0.5, 1.0 } },
        },
    };
    if (options.kind == .present) {
        const render_pass_info: vk.RenderPassBeginInfo = .{
            .render_pass = self.backend.render_pass,
            .framebuffer = self.backend.swapchain.framebufs.get(self.backend.image_index.?),
            .render_area = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.backend.swapchain.swap_extent,
            },
            .clear_value_count = clear_values.len,
            .p_clear_values = &clear_values,
        };
        self.backend.device.cmdBeginRenderPass(command_buffer, &render_pass_info, .@"inline");
    }

    return .{
        .cmds = .{
            .buf = .fromBackendType(command_buffer),
            .bindings = options.bindings,
            .tracy_queue = options.combined_queue.tracy_queue,
        },
        .queue = options.combined_queue.queue,
        .signal = options.signal,
        .zone = zone,
    };
}

pub fn zoneEnd(
    self: *Ctx,
    zone: Ctx.Zone,
    options: Ctx.Zone.EndOptions,
) void {
    if (self.backend.debug_messenger != .null_handle) {
        self.backend.device.cmdEndDebugUtilsLabelEXT(options.command_buffer.asBackendType());
    }

    if (tracy.enabled and self.backend.tracy_query_pool != .null_handle) {
        const end_query = zone.endId();
        self.backend.device.cmdWriteTimestamp(
            options.command_buffer.asBackendType(),
            .{ .bottom_of_pipe_bit = true },
            self.backend.tracy_query_pool,
            end_query,
        );
        options.tracy_queue.endZone(end_query);
    }
}

pub fn cmdBufGraphicsAppend(
    self: *Ctx,
    cmds: Ctx.Cmds(null),
    options: Ctx.Cmds(null).AppendGraphicsCmdsOptions,
) void {
    const command_buffer = cmds.buf.asBackendType();

    const zone = Ctx.Zone.begin(self, .{
        .command_buffer = cmds.buf,
        .tracy_queue = cmds.tracy_queue,
        .loc = options.loc,
    });
    defer zone.end(self, .{
        .command_buffer = cmds.buf,
        .tracy_queue = cmds.tracy_queue,
    });

    const bindings = cmds.bindings;

    for (options.cmds) |draw| {
        if (draw.combined_pipeline.pipeline != bindings.pipeline) {
            bindings.pipeline = draw.combined_pipeline.pipeline;

            // Bind the pipeline
            self.backend.device.cmdBindPipeline(
                command_buffer,
                .graphics,
                draw.combined_pipeline.pipeline.asBackendType(),
            );

            if (!bindings.dynamic_state) {
                bindings.dynamic_state = true;

                self.backend.device.cmdSetViewport(command_buffer, 0, 1, &.{.{
                    .x = 0.0,
                    .y = 0.0,
                    .width = @floatFromInt(self.backend.swapchain.swap_extent.width),
                    .height = @floatFromInt(self.backend.swapchain.swap_extent.height),
                    .min_depth = 0.0,
                    .max_depth = 1.0,
                }});

                self.backend.device.cmdSetScissor(command_buffer, 0, 1, &.{.{
                    .offset = .{ .x = 0, .y = 0 },
                    .extent = self.backend.swapchain.swap_extent,
                }});
            }
        }

        // Rebind the descriptor set if it changed. If we had multiple we'd have to invalidate any
        // descriptor sets following and including the last compatible one, but this isn't a concern
        // since we only support one: it's either the same and compatible or different and needs to
        // be rebound.
        if (draw.descriptor_set != bindings.descriptor_set) {
            bindings.descriptor_set = draw.descriptor_set;
            self.backend.device.cmdBindDescriptorSets(
                command_buffer,
                .graphics,
                draw.combined_pipeline.layout.asBackendType(),
                0,
                1,
                &.{draw.descriptor_set.asBackendType()},
                0,
                &[0]u32{},
            );
        }

        // Issue the draw call
        if (draw.indices) |indices| {
            // Rebind the index buffer if it has changed
            if (indices != bindings.indices) {
                bindings.indices = indices;
                comptime assert(Ctx.DrawCmd.IndexedIndirect.Index == u16); // Make sure we're binding it as the right type
                self.backend.device.cmdBindIndexBuffer(
                    command_buffer,
                    indices.asBackendType(),
                    0,
                    .uint16,
                );
            }

            self.backend.device.cmdDrawIndexedIndirect(
                command_buffer,
                draw.args.buf.asBackendType(),
                draw.args.offset,
                draw.args_count,
                @sizeOf(Ctx.DrawCmd.IndexedIndirect),
            );
        } else {
            self.backend.device.cmdDrawIndirect(
                command_buffer,
                draw.args.buf.asBackendType(),
                draw.args.offset,
                draw.args_count,
                @sizeOf(Ctx.DrawCmd.Indirect),
            );
        }
    }
}

pub fn combinedCmdBufSubmit(
    self: *Ctx,
    combined_command_buffer: Ctx.CombinedCmdBuf(null),
    kind: Ctx.CmdBufKind,
    wait: []const Ctx.Wait,
) void {
    const command_buffer = combined_command_buffer.cmds.buf.asBackendType();

    if (kind == .present) {
        self.backend.device.cmdEndRenderPass(command_buffer);
    }
    combined_command_buffer.zone.end(self, .{
        .command_buffer = combined_command_buffer.cmds.buf,
        .tracy_queue = combined_command_buffer.cmds.tracy_queue,
    });
    self.backend.device.endCommandBuffer(command_buffer) catch |err| @panic(@errorName(err));

    {
        const queue_submit_zone = Zone.begin(.{ .name = "queue submit", .src = @src() });
        defer queue_submit_zone.end();

        // Max is the number of command buffers, minus one since you can't wait on the final one,
        // plus one for the possibility of the image available semaphore.
        const max_waits = global_options.max_cmdbufs_per_frame;
        var wait_stages: std.BoundedArray(vk.PipelineStageFlags, max_waits) = .{};
        var wait_semaphores: std.BoundedArray(vk.Semaphore, max_waits) = .{};
        if (kind == .present) {
            wait_semaphores.appendAssumeCapacity(self.backend.image_availables[self.frameInFlight()]);
            wait_stages.appendAssumeCapacity(.{ .color_attachment_output_bit = true });
        }
        for (wait) |w| {
            wait_semaphores.appendAssumeCapacity(w.semaphore.asBackendType());
            wait_stages.appendAssumeCapacity(.{
                .top_of_pipe_bit = w.stages.top_of_pipe,
                .draw_indirect_bit = w.stages.draw_indirect,
                .vertex_input_bit = w.stages.vertex_input,
                .vertex_shader_bit = w.stages.vertex_shader,
                .tessellation_control_shader_bit = w.stages.tessellation_control_shader,
                .tessellation_evaluation_shader_bit = w.stages.tessellation_evaluation_shader,
                .geometry_shader_bit = w.stages.geometry_shader,
                .fragment_shader_bit = w.stages.fragment_shader,
                .early_fragment_tests_bit = w.stages.early_fragment_tests,
                .late_fragment_tests_bit = w.stages.late_fragment_tests,
                .color_attachment_output_bit = w.stages.color_attachment_output,
                .compute_shader_bit = w.stages.compute_shader,
                .transfer_bit = w.stages.transfer,
                .bottom_of_pipe_bit = w.stages.bottom_of_pipe,
                .host_bit = w.stages.host,
                .all_graphics_bit = w.stages.all_graphics,
                .all_commands_bit = w.stages.all_commands,
            });
        }

        const max_signals = 2;
        var signal_semaphores: std.BoundedArray(vk.Semaphore, max_signals) = .{};
        var signal_values: std.BoundedArray(u64, max_signals) = .{};

        signal_semaphores.appendAssumeCapacity(combined_command_buffer.signal.asBackendType());
        signal_values.appendAssumeCapacity(self.frame + self.frames_in_flight);

        if (kind == .present) {
            const ready_for_present = (self.backend.ready_for_present[self.frameInFlight()].addOne() catch @panic("OOB")).*;
            signal_semaphores.appendAssumeCapacity(ready_for_present);
            signal_values.appendAssumeCapacity(1); // Value ignored, not a timeline semaphore
        }

        const command_buffers = [_]vk.CommandBuffer{command_buffer};
        const timeline_semaphore_submit_info: vk.TimelineSemaphoreSubmitInfoKHR = .{
            .signal_semaphore_value_count = @intCast(signal_values.len),
            .p_signal_semaphore_values = &signal_values.buffer,
        };
        const submit_infos = [_]vk.SubmitInfo{.{
            .wait_semaphore_count = @intCast(wait_semaphores.len),
            .p_wait_semaphores = &wait_semaphores.buffer,
            .p_wait_dst_stage_mask = &wait_stages.buffer,
            .command_buffer_count = command_buffers.len,
            .p_command_buffers = &command_buffers,
            .signal_semaphore_count = @intCast(signal_semaphores.len),
            .p_signal_semaphores = &signal_semaphores.buffer,
            .p_next = &timeline_semaphore_submit_info,
        }};
        self.backend.device.queueSubmit(
            combined_command_buffer.queue.asBackendType(),
            submit_infos.len,
            &submit_infos,
            .null_handle,
        ) catch |err| @panic(@errorName(err));
    }
}

pub fn descriptorPoolDestroy(self: *Ctx, pool: Ctx.DescPool) void {
    self.backend.device.destroyDescriptorPool(pool.asBackendType(), null);
}

pub fn descriptorPoolReset(self: *Ctx, pool: Ctx.DescPool) void {
    self.backend.device.resetDescriptorPool(pool.asBackendType(), .{}) catch |err| @panic(@errorName(err));
}

pub fn descriptorPoolCreate(
    self: *Ctx,
    comptime max_cmds: u32,
    options: Ctx.DescPool.InitOptions,
) Ctx.DescPool {
    // Create a descriptor pool
    const descriptor_pool = b: {
        var uniform_buffers: u32 = 0;
        var storage_buffers: u32 = 0;
        var combined_image_samplers: u32 = 0;
        var descriptors: u32 = 0;

        for (options.cmds) |cmd| {
            for (cmd.layout_create_options.descriptors) |descriptor| {
                switch (descriptor.kind) {
                    .uniform_buffer => uniform_buffers += 1,
                    .storage_buffer => storage_buffers += 1,
                    .combined_image_sampler => combined_image_samplers += 1,
                }
            }
            descriptors += @intCast(cmd.layout_create_options.descriptors.len);
        }

        // Descriptor count must be greater than zero, so skip any that are zero
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorPoolSize.html
        var sizes: std.BoundedArray(vk.DescriptorPoolSize, 3) = .{};
        if (uniform_buffers > 0) sizes.appendAssumeCapacity(.{
            .type = .uniform_buffer,
            .descriptor_count = uniform_buffers,
        });
        if (storage_buffers > 0) sizes.appendAssumeCapacity(.{
            .type = .storage_buffer,
            .descriptor_count = storage_buffers,
        });
        if (combined_image_samplers > 0) sizes.appendAssumeCapacity(.{
            .type = .combined_image_sampler,
            .descriptor_count = combined_image_samplers,
        });

        const descriptor_pool = self.backend.device.createDescriptorPool(&.{
            .pool_size_count = @intCast(sizes.len),
            .p_pool_sizes = &sizes.buffer,
            .flags = .{},
            .max_sets = @intCast(options.cmds.len),
        }, null) catch |err| @panic(@errorName(err));
        setName(self.backend.device, descriptor_pool, options.name, self.backend.debug_messenger != .null_handle);

        break :b descriptor_pool;
    };

    // Create the descriptor sets
    {
        // Collect the arguments for descriptor set creation
        var layout_buf: std.BoundedArray(vk.DescriptorSetLayout, max_cmds) = .{};
        var results: [max_cmds]vk.DescriptorSet = undefined;
        for (options.cmds) |cmd| {
            layout_buf.appendAssumeCapacity(cmd.layout.asBackendType());
        }

        // Allocate the descriptor sets
        self.backend.device.allocateDescriptorSets(&.{
            .descriptor_pool = descriptor_pool,
            .descriptor_set_count = @intCast(layout_buf.len),
            .p_set_layouts = &layout_buf.buffer,
        }, &results) catch |err| @panic(@errorName(err));

        // Write the results
        for (options.cmds, results[0..options.cmds.len]) |cmd, result| {
            cmd.result.* = .fromBackendType(result);
            setName(self.backend.device, result, cmd.name, self.backend.debug_messenger != .null_handle);
        }
    }

    // Return the descriptor pool
    return .fromBackendType(descriptor_pool);
}

pub fn descriptorSetsUpdate(
    self: *Ctx,
    comptime max_updates: u32,
    updates: []const Ctx.DescUpdateCmd,
) void {
    var buffer_infos: std.BoundedArray(vk.DescriptorBufferInfo, max_updates) = .{};
    var combined_image_samplers: std.BoundedArray(vk.DescriptorImageInfo, max_updates) = .{};
    var write_sets: std.BoundedArray(vk.WriteDescriptorSet, max_updates) = .{};

    // Iterate over the updates
    var i: u32 = 0;
    while (i < updates.len) {
        // Find all subsequent updates on the same set binding and type
        const batch_first_update = updates[i];
        const batch_set = batch_first_update.set;
        const batch_binding = batch_first_update.binding;
        const batch_kind: Ctx.DescUpdateCmd.Value.Tag = batch_first_update.value;
        const batch_index_start: u32 = batch_first_update.index;
        var batch_size: u32 = 0;
        while (true) {
            const update_curr = updates[i + batch_size];

            switch (update_curr.value) {
                .uniform_buffer_view => |view| buffer_infos.appendAssumeCapacity(.{
                    .buffer = view.buf.asBackendType(),
                    .offset = view.offset,
                    .range = view.size,
                }),
                .storage_buffer_view => |view| buffer_infos.appendAssumeCapacity(.{
                    .buffer = view.buf.asBackendType(),
                    .offset = view.offset,
                    .range = view.size,
                }),
                .combined_image_sampler => {
                    const combined_image_sampler = update_curr.value.combined_image_sampler;
                    combined_image_samplers.appendAssumeCapacity(.{
                        .sampler = combined_image_sampler.sampler.asBackendType(),
                        .image_view = combined_image_sampler.view.asBackendType(),
                        .image_layout = layoutToVk(combined_image_sampler.layout),
                    });
                },
            }

            batch_size += 1;
            if (i + batch_size >= updates.len) break;
            const update_next = updates[i + batch_size];
            if (update_next.value != batch_kind or
                update_next.set != batch_set or
                update_next.binding != batch_binding or
                update_next.index != batch_index_start + batch_size) break;
        }

        // Write the update
        switch (batch_kind) {
            .uniform_buffer_view => {
                const batch_buffer_infos = buffer_infos.constSlice()[buffer_infos.len - batch_size ..];
                write_sets.appendAssumeCapacity(.{
                    .dst_set = batch_set.asBackendType(),
                    .dst_binding = batch_binding,
                    .dst_array_element = batch_index_start,
                    .descriptor_type = .uniform_buffer,
                    .descriptor_count = batch_size,
                    .p_buffer_info = batch_buffer_infos.ptr,
                    .p_image_info = &[0]vk.DescriptorImageInfo{},
                    .p_texel_buffer_view = &[0]vk.BufferView{},
                });
            },
            .storage_buffer_view => {
                const batch_buffer_infos = buffer_infos.constSlice()[buffer_infos.len - batch_size ..];
                write_sets.appendAssumeCapacity(.{
                    .dst_set = batch_set.asBackendType(),
                    .dst_binding = batch_binding,
                    .dst_array_element = batch_index_start,
                    .descriptor_type = .storage_buffer,
                    .descriptor_count = batch_size,
                    .p_buffer_info = batch_buffer_infos.ptr,
                    .p_image_info = &[0]vk.DescriptorImageInfo{},
                    .p_texel_buffer_view = &[0]vk.BufferView{},
                });
            },
            .combined_image_sampler => {
                const batch_combined_image_samplers = combined_image_samplers.constSlice()[combined_image_samplers.len - batch_size ..];
                write_sets.appendAssumeCapacity(.{
                    .dst_set = batch_set.asBackendType(),
                    .dst_binding = batch_binding,
                    .dst_array_element = batch_index_start,
                    .descriptor_type = .combined_image_sampler,
                    .descriptor_count = batch_size,
                    .p_buffer_info = &[0]vk.DescriptorBufferInfo{},
                    .p_image_info = batch_combined_image_samplers.ptr,
                    .p_texel_buffer_view = &[0]vk.BufferView{},
                });
            },
        }

        i += batch_size;
    }

    self.backend.device.updateDescriptorSets(@intCast(write_sets.len), &write_sets.buffer, 0, null);
}

pub fn acquireNextImage(self: *Ctx) ?u64 {
    var wait_timer = std.time.Timer.start() catch |err| @panic(@errorName(err));
    const aquire_result = b: {
        const acquire_zone = Zone.begin(.{
            .src = @src(),
            .name = "blocking: acquire next image",
            .color = global_options.blocking_zone_color,
        });
        defer acquire_zone.end();
        break :b self.backend.device.acquireNextImageKHR(
            self.backend.swapchain.swapchain,
            std.math.maxInt(u64),
            self.backend.image_availables[self.frameInFlight()],
            .null_handle,
        ) catch |err| switch (err) {
            error.OutOfDateKHR, error.FullScreenExclusiveModeLostEXT => {
                self.backend.swapchain.recreate(self);
                return null;
            },
            error.OutOfHostMemory,
            error.OutOfDeviceMemory,
            error.Unknown,
            error.SurfaceLostKHR,
            error.DeviceLost,
            => @panic(@errorName(err)),
        };
    };
    const wait_ns = wait_timer.lap();
    self.backend.image_index = aquire_result.image_index;
    return wait_ns;
}

pub fn frameStart(self: *Ctx) void {
    self.backend.image_index = null;

    const graphics_command_pool = self.backend.graphics_command_pools[self.frameInFlight()];
    self.backend.device.resetCommandPool(graphics_command_pool, .{}) catch |err| @panic(@errorName(err));

    const compute_command_pool = self.backend.compute_command_pools[self.frameInFlight()];
    self.backend.device.resetCommandPool(compute_command_pool, .{}) catch |err| @panic(@errorName(err));

    const transfer_command_pool = self.backend.transfer_command_pools[self.frameInFlight()];
    self.backend.device.resetCommandPool(transfer_command_pool, .{}) catch |err| @panic(@errorName(err));

    self.backend.ready_for_present[self.frameInFlight()].clear();

    if (self.backend.tracy_query_pool != .null_handle and self.backend.zones.handles.allocated > 0) {
        var results: [@as(usize, global_options.tracy_query_pool_capacity) * 2]u64 = undefined;
        const result = self.backend.device.getQueryPoolResults(
            self.backend.tracy_query_pool,
            0,
            global_options.tracy_query_pool_capacity,
            @as(usize, global_options.tracy_query_pool_capacity) * @sizeOf(u64) * 2,
            &results,
            @sizeOf(u64) * 2,
            .{
                .@"64_bit" = true,
                .with_availability_bit = true,
            },
        ) catch |err| @panic(@errorName(err));
        switch (result) {
            .success, .not_ready => {},
            else => @panic(@tagName(result)),
        }

        var i: u16 = 0;
        while (i < self.backend.zones.handles.allocated) : (i += 2) {
            const begin_index = i * 2;
            const end_index = i * 2 + 1;

            const begin_available = results[begin_index * 2 + 1] != 0;
            const end_available = results[end_index * 2 + 1] != 0;

            if (begin_available and end_available) {
                const begin_gpu_time = results[begin_index * 2];
                const end_gpu_time = results[end_index * 2];

                const tracy_queue = self.backend.zones.items[i];
                tracy_queue.emitTime(.{
                    .query_id = begin_index,
                    .gpu_time = begin_gpu_time,
                });
                tracy_queue.emitTime(.{
                    .query_id = end_index,
                    .gpu_time = end_gpu_time,
                });

                self.backend.device.resetQueryPool(self.backend.tracy_query_pool, @intCast(begin_index), 2);
                self.backend.zones.remove(@intCast(i));
            }
        }
    }
}

pub fn getDevice(self: *const @This()) Ctx.Device {
    return .{
        .kind = self.physical_device.ty,
        .uniform_buf_offset_alignment = self.physical_device.min_uniform_buffer_offset_alignment,
        .storage_buf_offset_alignment = self.physical_device.min_storage_buffer_offset_alignment,
        .timestamp_period = self.timestamp_period,
        .combined_queues = self.combined_queues,
    };
}

pub fn imageCreate(
    self: *Ctx,
    options: Ctx.ImageOptions,
) Ctx.Image(.{}) {
    const image = self.backend.device.createImage(&.{
        .flags = .{
            .sparse_binding_bit = options.flags.sparse_binding,
            .sparse_residency_bit = options.flags.sparse_residency,
            .sparse_aliased_bit = options.flags.sparse_aliased,
            .mutable_format_bit = options.flags.mutable_format,
            .cube_compatible_bit = options.flags.cube_compatible,
            .alias_bit = options.flags.alias,
            .split_instance_bind_regions_bit = options.flags.split_instance_bind_regions,
            .@"2d_array_compatible_bit" = options.flags.@"2d_array_compatible",
            .block_texel_view_compatible_bit = options.flags.block_texel_view_compatible,
            .extended_usage_bit = options.flags.extended_usage,
            .protected_bit = options.flags.protected,
        },
        .image_type = switch (options.dimensions) {
            .@"1d" => .@"1d",
            .@"2d" => .@"2d",
            .@"3d" => .@"3d",
        },
        .format = createImageOptionsFormatToVk(options.format),
        .extent = .{
            .width = options.extent.width,
            .height = options.extent.height,
            .depth = options.extent.depth,
        },
        .mip_levels = options.mip_levels,
        .array_layers = options.array_layers,
        .samples = switch (options.samples) {
            .@"1" => .{ .@"1_bit" = true },
            .@"2" => .{ .@"2_bit" = true },
            .@"4" => .{ .@"4_bit" = true },
            .@"8" => .{ .@"8_bit" = true },
            .@"16" => .{ .@"16_bit" = true },
            .@"32" => .{ .@"32_bit" = true },
            .@"64" => .{ .@"64_bit" = true },
        },
        .tiling = switch (options.tiling) {
            .optimal => .optimal,
            .linear => .linear,
        },
        .usage = .{
            .transfer_src_bit = options.usage.transfer_src,
            .transfer_dst_bit = options.usage.transfer_dst,
            .sampled_bit = options.usage.sampled,
            .storage_bit = options.usage.storage,
            .color_attachment_bit = options.usage.color_attachment,
            .depth_stencil_attachment_bit = options.usage.depth_stencil_attachment,
            .transient_attachment_bit = options.usage.transient_attachment,
            .input_attachment_bit = options.usage.input_attachment,
        },
        .sharing_mode = if (options.exclusive) .exclusive else .concurrent,
        .queue_family_index_count = 0,
        .p_queue_family_indices = null,
        .initial_layout = layoutToVk(options.initial_layout),
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.device, image, options.name, self.backend.debug_messenger != .null_handle);
    self.backend.device.bindImageMemory(
        image,
        options.location.memory.asBackendType(),
        options.location.offset,
    ) catch |err| @panic(@errorName(err));
    return .fromBackendType(image);
}

pub fn imageDestroy(self: *Ctx, image: Ctx.Image(.{})) void {
    self.backend.device.destroyImage(image.asBackendType(), null);
}

pub fn imageMemReqs(
    self: *Ctx,
    image: Ctx.Image(.{}),
) Ctx.MemReqs {
    const requirements = self.backend.device.getImageMemoryRequirements(image.asBackendType());
    return .{
        .size = requirements.size,
        .alignment = requirements.alignment,
    };
}

pub fn imageViewCreate(
    self: *Ctx,
    options: Ctx.ImageView.InitOptions,
) Ctx.ImageView {
    const image_view = self.backend.device.createImageView(&.{
        .image = options.image.asBackendType(),
        .view_type = switch (options.kind) {
            .@"1d" => .@"1d",
            .@"2d" => .@"2d",
            .@"3d" => .@"3d",
            .cube => .cube,
            .@"1d_array" => .@"1d_array",
            .@"2d_array" => .@"2d_array",
            .cube_array => .cube_array,
        },
        .format = createImageOptionsFormatToVk(options.format),
        .components = .{
            .r = swizzleToVk(options.components.r),
            .g = swizzleToVk(options.components.g),
            .b = swizzleToVk(options.components.b),
            .a = swizzleToVk(options.components.a),
        },
        .subresource_range = .{
            .aspect_mask = aspectToVk(options.aspect),
            .base_mip_level = options.base_mip_level,
            .level_count = options.level_count,
            .base_array_layer = options.base_array_layer,
            .layer_count = options.layer_count,
        },
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.device, image_view, options.name, self.backend.debug_messenger != .null_handle);
    return .fromBackendType(image_view);
}

pub fn imageViewDestroy(self: *Ctx, image_view: Ctx.ImageView) void {
    self.backend.device.destroyImageView(image_view.asBackendType(), null);
}

pub fn memoryCreate(
    self: *Ctx,
    options: Ctx.MemoryCreateUntypedOptions,
) Ctx.Memory(.{}) {
    // Determine the memory type and size. Vulkan requires that we create an image or buffer to be
    // able to do this, but we don't need to actually bind it to any memory it's just a handle.
    const memory_type_bits = switch (options.usage) {
        .color_image => |image| b: {
            // "For images created with a color format, the memoryTypeBits member is identical for
            // all VkImage objects created with the same combination of values for the tiling
            // member, the VK_IMAGE_CREATE_SPARSE_BINDING_BIT bit and VK_IMAGE_CREATE_PROTECTED_BIT
            // bit of the flags member, the VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT bit of
            // the flags member, the VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT bit of the usage member
            // if the VkPhysicalDeviceHostImageCopyPropertiesEXT::identicalMemoryTypeRequirements
            // property is VK_FALSE, handleTypes member of VkExternalMemoryImageCreateInfo, and the
            // VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT of the usage member in the VkImageCreateInfo
            // structure passed to vkCreateImage."
            var reqs: vk.MemoryRequirements2 = .{ .memory_requirements = undefined };
            self.backend.device.getDeviceImageMemoryRequirements(
                &.{
                    .p_create_info = &.{
                        .flags = .{},
                        .image_type = .@"2d",
                        .format = .r8g8b8a8_uint, // Supported by all DX12 hardware
                        .extent = .{
                            .width = 16,
                            .height = 16,
                            .depth = 1,
                        },
                        .mip_levels = 1,
                        .array_layers = 1,
                        .samples = .{ .@"1_bit" = true },
                        .tiling = switch (image.tiling) {
                            .optimal => .optimal,
                            .linear => .linear,
                        },
                        .usage = .{
                            .transient_attachment_bit = image.transient_attachment,
                            .sampled_bit = true,
                        },
                        .sharing_mode = .exclusive,
                        .queue_family_index_count = 0,
                        .p_queue_family_indices = null,
                        .initial_layout = .undefined,
                    },
                    .plane_aspect = .{},
                },
                &reqs,
            );
            const memory_type_bits: std.bit_set.IntegerBitSet(32) = .{
                .mask = reqs.memory_requirements.memory_type_bits,
            };
            break :b memory_type_bits;
        },
        .depth_stencil_image => |image| b: {
            // "For images created with a depth/stencil format, the memoryTypeBits member is
            // identical for all VkImage objects created with the same combination of values for the
            // format member, the tiling member, the VK_IMAGE_CREATE_SPARSE_BINDING_BIT bit and
            // VK_IMAGE_CREATE_PROTECTED_BIT bit of the flags member, the
            // VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT bit of the flags member, the
            // VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT bit of the usage member if the
            // VkPhysicalDeviceHostImageCopyPropertiesEXT::identicalMemoryTypeRequirements property
            // is VK_FALSE, handleTypes member of VkExternalMemoryImageCreateInfo, and the
            // VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT of the usage member in the VkImageCreateInfo
            // structure passed to vkCreateImage."
            var reqs: vk.MemoryRequirements2 = .{ .memory_requirements = undefined };
            self.backend.device.getDeviceImageMemoryRequirements(
                &.{
                    .p_create_info = &.{
                        .flags = .{},
                        .image_type = .@"2d",
                        .format = switch (image.format) {
                            .d24_unorm_s8_uint => .d24_unorm_s8_uint,
                        },
                        .extent = .{
                            .width = 16,
                            .height = 16,
                            .depth = 1,
                        },
                        .mip_levels = 1,
                        .array_layers = 1,
                        .samples = .{ .@"1_bit" = true },
                        .tiling = switch (image.tiling) {
                            .optimal => .optimal,
                            .linear => .linear,
                        },
                        .usage = .{
                            .transient_attachment_bit = image.transient_attachment,
                            .sampled_bit = true,
                        },
                        .sharing_mode = .exclusive,
                        .queue_family_index_count = 0,
                        .p_queue_family_indices = null,
                        .initial_layout = .undefined,
                    },
                    .plane_aspect = .{},
                },
                &reqs,
            );
            const memory_type_bits: std.bit_set.IntegerBitSet(32) = .{
                .mask = reqs.memory_requirements.memory_type_bits,
            };
            break :b memory_type_bits;
        },
    };

    const device_memory_properties = self.backend.instance.getPhysicalDeviceMemoryProperties(
        self.backend.physical_device.device,
    );
    const memory_type_index = findMemoryType(
        device_memory_properties,
        memory_type_bits,
        options.access,
    ) orelse @panic("unsupported memory type");

    // Allocate the memory
    const memory = self.backend.device.allocateMemory(&.{
        .allocation_size = options.size,
        .memory_type_index = memory_type_index,
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.device, memory, options.name, self.backend.debug_messenger != .null_handle);
    return .fromBackendType(memory);
}

pub fn deviceMemoryDestroy(self: *Ctx, memory: Ctx.Memory(.{})) void {
    self.backend.device.freeMemory(memory.asBackendType(), null);
}

pub fn combinedPipelineDestroy(self: *Ctx, combined: Ctx.CombinedPipeline(null)) void {
    self.backend.device.destroyPipeline(combined.pipeline.asBackendType(), null);
}

pub fn combinedPipelinesCreate(
    self: *Ctx,
    comptime max_cmds: u32,
    cmds: []const Ctx.InitCombinedPipelineCmd,
) void {
    // Other settings
    const dynamic_states = [_]vk.DynamicState{
        .viewport,
        .scissor,
    };
    const dynamic_state: vk.PipelineDynamicStateCreateInfo = .{
        .dynamic_state_count = dynamic_states.len,
        .p_dynamic_states = &dynamic_states,
    };

    const vertex_binding_descriptons = [_]vk.VertexInputBindingDescription{};
    const vertex_attribute_descriptons = [_]vk.VertexInputAttributeDescription{};
    const vertex_input_info: vk.PipelineVertexInputStateCreateInfo = .{
        .vertex_binding_description_count = vertex_binding_descriptons.len,
        .p_vertex_binding_descriptions = &vertex_binding_descriptons,
        .vertex_attribute_description_count = vertex_attribute_descriptons.len,
        .p_vertex_attribute_descriptions = &vertex_attribute_descriptons,
    };

    const viewport_state: vk.PipelineViewportStateCreateInfo = .{
        .viewport_count = 1,
        .scissor_count = 1,
    };

    const rasterizer: vk.PipelineRasterizationStateCreateInfo = .{
        .depth_clamp_enable = vk.FALSE,
        .rasterizer_discard_enable = vk.FALSE,
        .polygon_mode = .fill,
        .line_width = 1.0,
        .cull_mode = .{ .back_bit = true },
        .front_face = .clockwise,
        .depth_bias_enable = vk.FALSE,
        .depth_bias_constant_factor = 0.0,
        .depth_bias_clamp = 0.0,
        .depth_bias_slope_factor = 0.0,
    };

    const multisampling: vk.PipelineMultisampleStateCreateInfo = .{
        .sample_shading_enable = vk.FALSE,
        .rasterization_samples = .{ .@"1_bit" = true },
        .min_sample_shading = 1.0,
        .alpha_to_coverage_enable = vk.FALSE,
        .alpha_to_one_enable = vk.FALSE,
    };

    const color_blend_attachments = [_]vk.PipelineColorBlendAttachmentState{
        .{
            .color_write_mask = .{
                .r_bit = true,
                .g_bit = true,
                .b_bit = true,
                .a_bit = true,
            },
            .blend_enable = vk.TRUE,
            .src_color_blend_factor = .src_alpha,
            .dst_color_blend_factor = .one_minus_src_alpha,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
        },
    };

    const color_blending: vk.PipelineColorBlendStateCreateInfo = .{
        .logic_op_enable = vk.FALSE,
        .attachment_count = color_blend_attachments.len,
        .p_attachments = &color_blend_attachments,
        .logic_op = .copy,
        .blend_constants = .{ 0.0, 0.0, 0.0, 0.0 },
    };

    // Pipeline create info
    const max_shader_stages = Ctx.InitCombinedPipelineCmd.Stages.max_stages;
    var shader_stages_buf: std.BoundedArray(vk.PipelineShaderStageCreateInfo, max_cmds * max_shader_stages) = .{};
    defer for (shader_stages_buf.constSlice()) |stage| {
        self.backend.device.destroyShaderModule(stage.module, null);
    };
    var pipeline_infos: std.BoundedArray(vk.GraphicsPipelineCreateInfo, max_cmds) = .{};
    var input_assembly_buf: std.BoundedArray(vk.PipelineInputAssemblyStateCreateInfo, max_cmds) = .{};
    for (cmds) |cmd| {
        const input_assembly = input_assembly_buf.addOneAssumeCapacity();
        input_assembly.* = switch (cmd.input_assembly) {
            .point_list => .{
                .topology = .point_list,
                .primitive_restart_enable = vk.FALSE,
            },
            .line_list => .{
                .topology = .line_list,
                .primitive_restart_enable = vk.FALSE,
            },
            .line_strip => |opt| .{
                .topology = .line_strip,
                .primitive_restart_enable = @intFromBool(opt.indexed_primitive_restart),
            },
            .triangle_list => .{
                .topology = .triangle_list,
                .primitive_restart_enable = vk.FALSE,
            },
            .triangle_strip => |opt| .{
                .topology = .triangle_strip,
                .primitive_restart_enable = @intFromBool(opt.indexed_primitive_restart),
            },
            .line_list_with_adjacency => .{
                .topology = .line_list_with_adjacency,
                .primitive_restart_enable = vk.FALSE,
            },
            .line_strip_with_adjacency => |opt| .{
                .topology = .line_strip_with_adjacency,
                .primitive_restart_enable = @intFromBool(opt.indexed_primitive_restart),
            },
            .triangle_list_with_adjacency => .{
                .topology = .triangle_list_with_adjacency,
                .primitive_restart_enable = vk.FALSE,
            },
            .triangle_strip_with_adjacency => |opt| .{
                .topology = .triangle_strip_with_adjacency,
                .primitive_restart_enable = @intFromBool(opt.indexed_primitive_restart),
            },
            .patch_list => .{
                .topology = .patch_list,
                .primitive_restart_enable = vk.FALSE,
            },
        };
        const vertex_module = self.backend.device.createShaderModule(&.{
            .code_size = cmd.stages.vertex.spv.len * @sizeOf(u32),
            .p_code = cmd.stages.vertex.spv.ptr,
        }, null) catch |err| @panic(@errorName(err));
        setName(self.backend.device, vertex_module, cmd.stages.vertex.name, self.backend.debug_messenger != .null_handle);

        shader_stages_buf.appendAssumeCapacity(.{
            .stage = .{ .vertex_bit = true },
            .module = vertex_module,
            .p_name = "main",
        });

        const fragment_module = self.backend.device.createShaderModule(&.{
            .code_size = cmd.stages.fragment.spv.len * @sizeOf(u32),
            .p_code = cmd.stages.fragment.spv.ptr,
        }, null) catch |err| @panic(@errorName(err));
        setName(self.backend.device, fragment_module, cmd.stages.fragment.name, self.backend.debug_messenger != .null_handle);

        shader_stages_buf.appendAssumeCapacity(.{
            .stage = .{ .fragment_bit = true },
            .module = fragment_module,
            .p_name = "main",
        });

        const shader_stages = shader_stages_buf.constSlice()[shader_stages_buf.len - 2 ..];
        pipeline_infos.appendAssumeCapacity(.{
            .stage_count = max_shader_stages,
            .p_stages = shader_stages.ptr,
            .p_vertex_input_state = &vertex_input_info,
            .p_input_assembly_state = input_assembly,
            .p_viewport_state = &viewport_state,
            .p_rasterization_state = &rasterizer,
            .p_multisample_state = &multisampling,
            .p_depth_stencil_state = null,
            .p_color_blend_state = &color_blending,
            .p_dynamic_state = &dynamic_state,
            .layout = cmd.layout.pipeline.asBackendType(),
            .render_pass = self.backend.render_pass,
            .subpass = 0,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = -1,
        });
    }

    // Create the pipelines
    var pipelines: [max_cmds]vk.Pipeline = undefined;
    const create_result = self.backend.device.createGraphicsPipelines(
        .null_handle,
        @intCast(pipeline_infos.len),
        &pipeline_infos.buffer,
        null,
        &pipelines,
    ) catch |err| @panic(@errorName(err));
    switch (create_result) {
        .success => {},
        else => |err| @panic(@tagName(err)),
    }
    for (pipelines[0..cmds.len], cmds) |pipeline, cmd| {
        setName(self.backend.device, pipeline, cmd.pipeline_name, self.backend.debug_messenger != .null_handle);
        cmd.result.* = .{
            .layout = cmd.layout.pipeline,
            .pipeline = .fromBackendType(pipeline),
        };
    }
}

pub fn present(self: *Ctx, queue: Ctx.Queue(.graphics)) u64 {
    const suboptimal_out_of_date, const present_ns = b: {
        const queue_present_zone = Zone.begin(.{ .name = "queue present", .src = @src() });
        defer queue_present_zone.end();
        const swapchain = [_]vk.SwapchainKHR{self.backend.swapchain.swapchain};
        const image_index = [_]u32{self.backend.image_index.?};
        const wait_semaphores = self.backend.ready_for_present[self.frameInFlight()].constSlice();
        const present_info: vk.PresentInfoKHR = .{
            .wait_semaphore_count = @intCast(wait_semaphores.len),
            .p_wait_semaphores = wait_semaphores.ptr,
            .swapchain_count = swapchain.len,
            .p_swapchains = &swapchain,
            .p_image_indices = &image_index,
            .p_results = null,
        };

        var present_timer = std.time.Timer.start() catch |err| @panic(@errorName(err));
        {
            const blocking_zone = Zone.begin(.{
                .src = @src(),
                .name = "blocking: queue present",
                .color = global_options.blocking_zone_color,
            });
            defer blocking_zone.end();

            const suboptimal_out_of_date = if (self.backend.device.queuePresentKHR(
                queue.asBackendType(),
                &present_info,
            )) |result|
                result == .suboptimal_khr
            else |err| switch (err) {
                error.OutOfDateKHR, error.FullScreenExclusiveModeLostEXT => true,
                error.OutOfHostMemory,
                error.OutOfDeviceMemory,
                error.Unknown,
                error.SurfaceLostKHR,
                error.DeviceLost,
                => @panic(@errorName(err)),
            };
            break :b .{ suboptimal_out_of_date, present_timer.lap() };
        }
    };

    const framebuffer_size_changed = !std.meta.eql(self.backend.swapchain.external_framebuf_size, self.framebuf_size);

    if (suboptimal_out_of_date or framebuffer_size_changed) {
        self.backend.swapchain.recreate(self);
    }

    return present_ns;
}

pub fn samplerCreate(
    self: *Ctx,
    options: Ctx.Sampler.InitOptions,
) Ctx.Sampler {
    const sampler = self.backend.device.createSampler(&.{
        .mag_filter = filterToVk(options.mag_filter),
        .min_filter = filterToVk(options.min_filter),
        .mipmap_mode = switch (options.mipmap_mode) {
            .nearest => .nearest,
            .linear => .linear,
        },
        .address_mode_u = addressModeToVk(options.address_mode.u),
        .address_mode_v = addressModeToVk(options.address_mode.v),
        .address_mode_w = addressModeToVk(options.address_mode.w),
        .mip_lod_bias = options.mip_lod_bias,
        .anisotropy_enable = @intFromBool(options.max_anisotropy_hint != 0.0 and self.backend.physical_device.sampler_anisotropy),
        .max_anisotropy = @min(options.max_anisotropy_hint, self.backend.physical_device.max_sampler_anisotropy),
        .compare_enable = @intFromBool(options.compare_op != null),
        .compare_op = if (options.compare_op) |compare_op| switch (compare_op) {
            .never => .never,
            .less => .less,
            .equal => .equal,
            .less_or_equal => .less_or_equal,
            .greater => .greater,
            .not_equal => .not_equal,
            .greater_or_equal => .greater_or_equal,
            .always => .always,
        } else .never,
        .min_lod = options.min_lod,
        .max_lod = options.max_lod orelse vk.LOD_CLAMP_NONE,
        .border_color = switch (options.border_color) {
            .float_transparent_black => .float_transparent_black,
            .int_transparent_black => .int_transparent_black,
            .float_opaque_black => .float_opaque_black,
            .int_opaque_black => .int_opaque_black,
            .float_opaque_white => .float_opaque_white,
            .int_opaque_white => .int_opaque_white,
        },
        .unnormalized_coordinates = @intFromBool(options.unnormalized_coordinates),
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.device, sampler, options.name, self.backend.debug_messenger != .null_handle);
    return .fromBackendType(sampler);
}

pub fn samplerDestroy(self: *Ctx, sampler: Ctx.Sampler) void {
    self.backend.device.destroySampler(sampler.asBackendType(), null);
}

pub fn timestampCalibration(self: *Ctx) Ctx.TimestampCalibration {
    return timestampCalibrationImpl(self.backend.device, self.backend.timestamp_queries);
}

pub fn semaphoreCreate(self: *Ctx, initial_value: u64) Ctx.Semaphore {
    const semaphore_type_create_info: vk.SemaphoreTypeCreateInfo = .{
        .semaphore_type = .timeline,
        .initial_value = initial_value,
    };

    const create_info: vk.SemaphoreCreateInfo = .{
        .p_next = &semaphore_type_create_info,
        .flags = .{},
    };

    const semaphore = self.backend.device.createSemaphore(
        &create_info,
        null,
    ) catch |err| @panic(@errorName(err));

    return .fromBackendType(semaphore);
}

pub fn semaphoreDestroy(self: *Ctx, semaphore: Ctx.Semaphore) void {
    self.backend.device.destroySemaphore(semaphore.asBackendType(), null);
}

pub fn semaphoreSignal(self: *Ctx, semaphore: Ctx.Semaphore, value: u64) void {
    self.backend.device.signalSemaphore(&.{
        .semaphore = semaphore.asBackendType(),
        .value = value,
    }) catch |err| @panic(@errorName(err));
}

pub fn semaphoresWait(self: *Ctx, semaphores: []const Ctx.Semaphore, values: []const u64) void {
    assert(semaphores.len == values.len);
    if (semaphores.len == 0) return;
    const result = self.backend.device.waitSemaphores(&.{
        .semaphore_count = @intCast(semaphores.len),
        .p_semaphores = @ptrCast(semaphores.ptr),
        .p_values = values.ptr,
        .flags = .{},
    }, std.math.maxInt(u64)) catch |err| @panic(@errorName(err));
    if (result != .success) @panic(@tagName(result));
}

pub fn timestampCalibrationImpl(
    device: vk.DeviceProxy,
    timestamp_queries: bool,
) Ctx.TimestampCalibration {
    if (!timestamp_queries) return .{
        .cpu = 0,
        .gpu = 0,
        .max_deviation = 0,
    };
    var calibration_results: [2]u64 = undefined;
    const max_deviation = device.getCalibratedTimestampsKHR(
        2,
        &.{
            .{ .time_domain = .clock_monotonic_raw_khr },
            .{ .time_domain = .device_khr },
        },
        &calibration_results,
    ) catch |err| @panic(@errorName(err));
    return .{
        .cpu = calibration_results[0],
        .gpu = calibration_results[1],
        .max_deviation = max_deviation,
    };
}

pub fn cmdBufTransferAppend(
    self: *Ctx,
    cmds: Ctx.Cmds(null),
    comptime max_regions: u32,
    options: Ctx.Cmds(null).AppendTransferCmdsOptions,
) void {
    const command_buffer = cmds.buf.asBackendType();

    const zone = Ctx.Zone.begin(self, .{
        .command_buffer = .fromBackendType(command_buffer),
        .tracy_queue = cmds.tracy_queue,
        .loc = options.loc,
    });

    for (options.cmds) |cmd| {
        switch (cmd) {
            .copy_buffer_to_color_image => |cmd_options| {
                // Transition the image to transfer dst optimal
                self.backend.device.cmdPipelineBarrier(
                    command_buffer,
                    .{ .top_of_pipe_bit = true },
                    .{ .transfer_bit = true },
                    .{},
                    0,
                    null,
                    0,
                    null,
                    1,
                    &.{.{
                        .src_access_mask = .{},
                        .dst_access_mask = .{ .transfer_write_bit = true },
                        .old_layout = .undefined,
                        .new_layout = .transfer_dst_optimal,
                        .src_queue_family_index = 0, // Ignored
                        .dst_queue_family_index = 0, // Ignored
                        .image = cmd_options.image.asBackendType(),
                        .subresource_range = .{
                            // Note: "If the queue family used to create the VkCommandPool which
                            // CmdBuf was allocated from does not support
                            // VK_QUEUE_GRAPHICS_BIT, for each element of pRegions, the aspectMask
                            // member of imageSubresource must not be VK_IMAGE_ASPECT_DEPTH_BIT or
                            // VK_IMAGE_ASPECT_STENCIL_BIT"
                            //
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdCopyBufferToImage.html
                            .aspect_mask = .{ .color_bit = true },
                            .base_mip_level = cmd_options.base_mip_level,
                            .level_count = cmd_options.level_count,
                            .base_array_layer = cmd_options.base_array_layer,
                            .layer_count = cmd_options.layer_count,
                        },
                    }},
                );

                // Copy data from the buffer to the image
                var regions: std.BoundedArray(vk.BufferImageCopy, max_regions) = .{};
                for (cmd_options.regions) |region| {
                    regions.append(.{
                        .buffer_offset = region.buffer_offset,
                        .buffer_row_length = region.buffer_row_length orelse 0,
                        .buffer_image_height = region.buffer_image_height orelse 0,
                        .image_subresource = .{
                            .aspect_mask = .{ .color_bit = true },
                            .mip_level = region.mip_level,
                            .base_array_layer = region.base_array_layer,
                            .layer_count = region.layer_count,
                        },
                        .image_offset = .{
                            .x = region.image_offset.x,
                            .y = region.image_offset.y,
                            .z = region.image_offset.z,
                        },
                        .image_extent = .{
                            .width = region.image_extent.width,
                            .height = region.image_extent.height,
                            .depth = region.image_extent.depth,
                        },
                    }) catch @panic("OOB");
                }
                self.backend.device.cmdCopyBufferToImage(
                    command_buffer,
                    cmd_options.buf.asBackendType(),
                    cmd_options.image.asBackendType(),
                    .transfer_dst_optimal,
                    @intCast(regions.len),
                    &regions.buffer,
                );

                // Transition to the destination layout
                self.backend.device.cmdPipelineBarrier(
                    command_buffer,
                    .{ .transfer_bit = true },
                    .{ .transfer_bit = true },
                    .{},
                    0,
                    null,
                    0,
                    null,
                    1,
                    &.{.{
                        .src_access_mask = .{ .transfer_write_bit = true },
                        .dst_access_mask = .{ .transfer_write_bit = true },
                        .old_layout = .transfer_dst_optimal,
                        .new_layout = layoutToVk(cmd_options.new_layout),
                        .src_queue_family_index = 0, // Ignored
                        .dst_queue_family_index = 0, // Ignored
                        .image = cmd_options.image.asBackendType(),
                        .subresource_range = .{
                            .aspect_mask = .{ .color_bit = true },
                            .base_mip_level = cmd_options.base_mip_level,
                            .level_count = cmd_options.level_count,
                            .base_array_layer = cmd_options.base_array_layer,
                            .layer_count = cmd_options.layer_count,
                        },
                    }},
                );
            },
            .copy_buffer_to_buffer => |cmd_options| {
                var regions: std.BoundedArray(vk.BufferCopy, max_regions) = .{};
                for (cmd_options.regions) |region| {
                    regions.append(.{
                        .src_offset = region.src_offset,
                        .dst_offset = region.dst_offset,
                        .size = region.size,
                    }) catch @panic("OOB");
                }
                self.backend.device.cmdCopyBuffer(
                    command_buffer,
                    cmd_options.src.asBackendType(),
                    cmd_options.dst.asBackendType(),
                    @intCast(regions.len),
                    &regions.buffer,
                );
            },
        }
    }

    zone.end(self, .{
        .command_buffer = .fromBackendType(command_buffer),
        .tracy_queue = cmds.tracy_queue,
    });
}

pub fn waitIdle(self: *const Ctx) void {
    self.backend.device.deviceWaitIdle() catch |err| @panic(@errorName(err));
}

fn layoutToVk(layout: Ctx.ImageOptions.Layout) vk.ImageLayout {
    return switch (layout) {
        .undefined => .undefined,
        .general => .general,
        .color_attachment_optimal => .color_attachment_optimal,
        .depth_stencil_attachment_optimal => .depth_stencil_attachment_optimal,
        .depth_stencil_read_only_optimal => .depth_stencil_read_only_optimal,
        .shader_read_only_optimal => .shader_read_only_optimal,
        .transfer_src_optimal => .transfer_src_optimal,
        .transfer_dst_optimal => .transfer_dst_optimal,
        .preinitialized => .preinitialized,
        .depth_read_only_stencil_attachment_optimal => .depth_read_only_stencil_attachment_optimal,
        .depth_attachment_stencil_read_only_optimal => .depth_attachment_stencil_read_only_optimal,
        .depth_attachment_optimal => .depth_attachment_optimal,
        .depth_read_only_optimal => .depth_read_only_optimal,
        .stencil_attachment_optimal => .stencil_attachment_optimal,
        .stencil_read_only_optimal => .stencil_read_only_optimal,
        .read_only_optimal => .read_only_optimal,
        .attachment_optimal => .attachment_optimal,
    };
}

fn createImageOptionsFormatToVk(format: Ctx.ImageOptions.Format) vk.Format {
    return switch (format) {
        .color => |color| switch (color) {
            .r8g8b8a8_srgb => .r8g8b8a8_srgb,
        },
        .depth_stencil => |depth_stencil_format| switch (depth_stencil_format) {
            .d24_unorm_s8_uint => .d24_unorm_s8_uint,
        },
    };
}

fn swizzleToVk(
    swizzle: Ctx.ImageView.InitOptions.ComponentMapping.Swizzle,
) vk.ComponentSwizzle {
    return switch (swizzle) {
        .identity => .identity,
        .zero => .zero,
        .one => .one,
        .r => .r,
        .g => .g,
        .b => .b,
        .a => .a,
    };
}

fn aspectToVk(self: Ctx.ImageView.InitOptions.Aspect) vk.ImageAspectFlags {
    return .{
        .color_bit = self.color,
        .depth_bit = self.depth,
        .stencil_bit = self.stencil,
        .metadata_bit = self.metadata,
        .plane_0_bit = self.plane_0,
        .plane_1_bit = self.plane_1,
        .plane_2_bit = self.plane_2,
    };
}

fn filterToVk(filter: Ctx.Sampler.InitOptions.Filter) vk.Filter {
    return switch (filter) {
        .nearest => .nearest,
        .linear => .linear,
    };
}

fn addressModeToVk(mode: Ctx.Sampler.InitOptions.AddressMode) vk.SamplerAddressMode {
    return switch (mode) {
        .repeat => .repeat,
        .mirrored_repeat => .mirrored_repeat,
        .clamp_to_edge => .clamp_to_edge,
        .clamp_to_border => .clamp_to_border,
        .mirror_clamp_to_edge => .mirror_clamp_to_edge,
    };
}

fn findMemoryType(
    device_memory_properties: vk.PhysicalDeviceMemoryProperties,
    required_type_bits: std.bit_set.IntegerBitSet(32),
    access: Ctx.MemoryCreateUntypedOptions.Access,
) ?u32 {
    const host_visible = switch (access) {
        .read, .write => true,
        .none => false,
    };
    const required_props: vk.MemoryPropertyFlags = .{
        .device_local_bit = switch (access) {
            .none => true,
            .read => false,
            .write => |write| write.prefer_device_local,
        },
        .host_visible_bit = host_visible,
        .host_coherent_bit = host_visible,
        .host_cached_bit = access == .read,
    };

    for (0..device_memory_properties.memory_type_count) |i| {
        // Check if this type is supported
        if (!required_type_bits.isSet(i)) continue;

        // Check for the required and forbidden memory property flags
        const found_mem_flags_i: u32 = @bitCast(device_memory_properties.memory_types[i].property_flags);
        const required_mem_flags_i: u32 = @bitCast(required_props);
        if (required_mem_flags_i & found_mem_flags_i != required_mem_flags_i) continue;

        // We passed all the checks, and Vulkan guarantees faster memory types are earlier in the
        // list, so pick this type!
        return @intCast(i);
    }

    switch (access) {
        .write => |write| if (write.prefer_device_local) {
            return findMemoryType(
                device_memory_properties,
                required_type_bits,
                .{ .write = .{ .prefer_device_local = false } },
            );
        },
        else => {},
    }

    return null;
}

const Swapchain = struct {
    // In practice, there should typically be only two or three.
    const max_swapchain_depth = 8;

    swapchain: vk.SwapchainKHR,
    images: std.BoundedArray(vk.Image, max_swapchain_depth),
    views: std.BoundedArray(vk.ImageView, max_swapchain_depth),
    framebufs: std.BoundedArray(vk.Framebuffer, max_swapchain_depth),
    swap_extent: vk.Extent2D,
    external_framebuf_size: struct { u32, u32 },

    pub fn init(
        instance: vk.InstanceProxy,
        framebuf_size: struct { u32, u32 },
        device: vk.DeviceProxy,
        physical_device: PhysicalDevice,
        surface: vk.SurfaceKHR,
        render_pass: vk.RenderPass,
        old_swapchain: vk.SwapchainKHR,
        validation: bool,
    ) @This() {
        const zone = tracy.Zone.begin(.{ .name = "swapchain init", .src = @src() });
        defer zone.end();

        const surface_capabilities = instance.getPhysicalDeviceSurfaceCapabilitiesKHR(
            physical_device.device,
            surface,
        ) catch |err| @panic(@errorName(err));
        const swap_extent = if (surface_capabilities.current_extent.width == std.math.maxInt(u32) and
            surface_capabilities.current_extent.height == std.math.maxInt(u32))
        e: {
            const width, const height = framebuf_size;
            break :e vk.Extent2D{
                .width = std.math.clamp(
                    width,
                    surface_capabilities.min_image_extent.width,
                    surface_capabilities.max_image_extent.width,
                ),
                .height = std.math.clamp(
                    height,
                    surface_capabilities.min_image_extent.height,
                    surface_capabilities.max_image_extent.height,
                ),
            };
        } else surface_capabilities.current_extent;

        const max_images = if (surface_capabilities.max_image_count == 0) b: {
            break :b std.math.maxInt(u32);
        } else surface_capabilities.max_image_count;
        const min_image_count = @min(max_images, surface_capabilities.min_image_count + 1);
        var swapchain_create_info: vk.SwapchainCreateInfoKHR = .{
            .surface = surface,
            .min_image_count = min_image_count,
            .image_format = physical_device.surface_format.format,
            .image_color_space = physical_device.surface_format.color_space,
            .image_extent = swap_extent,
            .image_array_layers = 1,
            .image_usage = .{ .color_attachment_bit = true },
            .pre_transform = surface_capabilities.current_transform,
            .composite_alpha = physical_device.composite_alpha,
            .present_mode = physical_device.present_mode,
            .clipped = vk.TRUE,
            .image_sharing_mode = .exclusive,
            .p_queue_family_indices = null,
            .queue_family_index_count = 0,
            .old_swapchain = old_swapchain,
        };

        const swapchain = device.createSwapchainKHR(&swapchain_create_info, null) catch |err| @panic(@errorName(err));
        setName(device, swapchain, .{ .str = "Main" }, validation);
        var images: std.BoundedArray(vk.Image, max_swapchain_depth) = .{};
        var image_count: u32 = max_swapchain_depth;
        const get_images_result = device.getSwapchainImagesKHR(
            swapchain,
            &image_count,
            &images.buffer,
        ) catch |err| @panic(@errorName(err));
        if (get_images_result != .success) @panic(@tagName(get_images_result));
        images.len = image_count;
        var views: std.BoundedArray(vk.ImageView, max_swapchain_depth) = .{};
        for (images.constSlice(), 0..) |image, i| {
            setName(device, image, .{ .str = "Swapchain", .index = i }, validation);
            const create_info: vk.ImageViewCreateInfo = .{
                .image = image,
                .view_type = .@"2d",
                .format = physical_device.surface_format.format,
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .base_mip_level = 0,
                    .level_count = 1,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .components = .{
                    .r = .identity,
                    .g = .identity,
                    .b = .identity,
                    .a = .identity,
                },
            };
            const view = device.createImageView(&create_info, null) catch |err| @panic(@errorName(err));
            setName(device, view, .{ .str = "Swapchain", .index = i }, validation);
            views.appendAssumeCapacity(view);
        }

        var framebufs: std.BoundedArray(vk.Framebuffer, max_swapchain_depth) = .{};
        for (views.constSlice(), 0..) |view, i| {
            const attachments = [1]vk.ImageView{view};

            const framebuffer_info: vk.FramebufferCreateInfo = .{
                .render_pass = render_pass,
                .attachment_count = attachments.len,
                .p_attachments = &attachments,
                .width = swap_extent.width,
                .height = swap_extent.height,
                .layers = 1,
            };

            const framebuf = device.createFramebuffer(&framebuffer_info, null) catch |err| @panic(@errorName(err));
            setName(device, framebuf, .{ .str = "Swapchain", .index = i }, validation);
            framebufs.appendAssumeCapacity(framebuf);
        }

        return .{
            .swapchain = swapchain,
            .images = images,
            .views = views,
            .framebufs = framebufs,
            .swap_extent = swap_extent,
            .external_framebuf_size = framebuf_size,
        };
    }

    pub fn destroyEverythingExceptSwapchain(self: *@This(), device: vk.DeviceProxy) void {
        for (self.framebufs.constSlice()) |f| device.destroyFramebuffer(f, null);
        for (self.views.constSlice()) |v| device.destroyImageView(v, null);
    }

    pub fn deinit(self: *@This(), device: vk.DeviceProxy) void {
        self.destroyEverythingExceptSwapchain(device);
        device.destroySwapchainKHR(self.swapchain, null);
        self.* = undefined;
    }

    pub fn recreate(self: *@This(), gx: *Ctx) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        // We wait idle on every recreate so that we can delete the old swapchain after. Technically
        // this is still incorrect, there is a spec bug that makes it impossible to do this
        // correctly without an extension that isn't widely supported:
        //
        // https://github.com/KhronosGroup/Vulkan-Docs/issues/1678
        //
        // If this causes us issues, and the extension still isn't widely supported, we can queue up
        // retired swapchains and wait a few seconds before deleting them or something.
        gx.backend.device.deviceWaitIdle() catch |err| std.debug.panic("vkDeviceWaitIdle failed: {}", .{err});
        const retired = self.swapchain;
        self.destroyEverythingExceptSwapchain(gx.backend.device);
        self.* = .init(
            gx.backend.instance,
            gx.framebuf_size,
            gx.backend.device,
            gx.backend.physical_device,
            gx.backend.surface,
            gx.backend.render_pass,
            retired,
            gx.backend.debug_messenger != .null_handle,
        );
        gx.backend.device.destroySwapchainKHR(retired, null);
    }
};

const CommandBufferSemaphores = std.BoundedArray(vk.Semaphore, global_options.max_cmdbufs_per_frame);

// r: clean up...undefined, and way it's used, etc
const PhysicalDevice = struct {
    device: vk.PhysicalDevice = .null_handle,
    name: [vk.MAX_PHYSICAL_DEVICE_NAME_SIZE]u8 = .{0} ** vk.MAX_PHYSICAL_DEVICE_NAME_SIZE,
    index: usize = std.math.maxInt(usize),
    rank: u8 = 0,
    surface_format: vk.SurfaceFormatKHR = undefined,
    present_mode: vk.PresentModeKHR = undefined,
    swap_extent: vk.Extent2D = undefined,
    surface_capabilities: vk.SurfaceCapabilitiesKHR = undefined,
    composite_alpha: vk.CompositeAlphaFlagsKHR = undefined,
    ty: Ctx.Device.Kind = undefined,
    min_uniform_buffer_offset_alignment: u16 = undefined,
    min_storage_buffer_offset_alignment: u16 = undefined,
    max_draw_indirect_count: u32 = undefined,
    sampler_anisotropy: bool = undefined,
    max_sampler_anisotropy: f32 = undefined,
};

fn queueFamilyHasPresent(
    instance: vk.InstanceProxy,
    device: vk.PhysicalDevice,
    surface: vk.SurfaceKHR,
    qfi: u32,
) bool {
    return (instance.getPhysicalDeviceSurfaceSupportKHR(
        device,
        @intCast(qfi),
        surface,
    ) catch |err| @panic(@errorName(err))) == vk.TRUE;
}

fn queueFamilyHasGraphics(properties: vk.QueueFamilyProperties) bool {
    return properties.queue_flags.graphics_bit;
}

fn queueFamilyHasCompute(properties: vk.QueueFamilyProperties) bool {
    return properties.queue_flags.compute_bit;
}

fn queueFamilyHasTransfer(properties: vk.QueueFamilyProperties) bool {
    return properties.queue_flags.transfer_bit or
        queueFamilyHasGraphics(properties) or
        queueFamilyHasCompute(properties);
}

const TimestampQueries = struct {
    pool: vk.QueryPool,
    count: u32,
    read: u32 = 0,
    write: u32 = 0,
};

const enabled_validation_features = [_]vk.ValidationFeatureEnableEXT{
    .gpu_assisted_ext,
    .gpu_assisted_reserve_binding_slot_ext,
    .best_practices_ext,
    .synchronization_validation_ext,
};

const validation_features: vk.ValidationFeaturesEXT = .{
    .enabled_validation_feature_count = enabled_validation_features.len,
    .p_enabled_validation_features = &enabled_validation_features,
    .p_next = &create_debug_messenger_info,
};

const create_debug_messenger_info: vk.DebugUtilsMessengerCreateInfoEXT = .{
    .message_severity = .{
        .verbose_bit_ext = true,
        .warning_bit_ext = true,
        .error_bit_ext = true,
    },
    .message_type = .{
        .general_bit_ext = true,
        .validation_bit_ext = true,
        .performance_bit_ext = true,
    },
    .pfn_user_callback = &vkDebugCallback,
};

const FormatDebugMessageData = struct {
    severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk.DebugUtilsMessageTypeFlagsEXT,
    data: [*c]const vk.DebugUtilsMessengerCallbackDataEXT,
    user_data: ?*anyopaque,
};

fn formatDebugMessage(
    data: FormatDebugMessageData,
    comptime _: []const u8,
    _: std.fmt.FormatOptions,
    writer: anytype,
) !void {
    try writer.writeAll("vulkan debug message:\n");
    try writer.print("  * type: {}\n", .{data.message_type});

    if (data.data) |d| {
        try writer.writeAll("  * id: ");
        if (d.*.p_message_id_name) |name| {
            try writer.print("{s}", .{name});
        } else {
            try writer.writeAll("null");
        }
        try writer.print(" ({})\n", .{d.*.message_id_number});

        if (d.*.p_message) |message| {
            try writer.print("  * message: {s}", .{message});
        }

        if (d.*.queue_label_count > 0) {
            try writer.writeByte('\n');
            if (d.*.p_queue_labels) |queue_labels| {
                for (queue_labels[0..d.*.queue_label_count]) |label| {
                    try writer.print("    * queue: {s}\n", .{label.p_label_name});
                }
            }
        }

        if (d.*.cmd_buf_label_count > 0) {
            try writer.writeByte('\n');
            if (d.*.p_cmd_buf_labels) |cmd_buf_labels| {
                for (cmd_buf_labels[0..d.*.cmd_buf_label_count]) |label| {
                    try writer.print("    * command buffer: {s}\n", .{label.p_label_name});
                }
            }
        }

        if (d.*.object_count > 0) {
            try writer.writeByte('\n');
            if (d.*.p_objects) |objects| {
                for (objects[0..d.*.object_count], 0..) |object, object_i| {
                    try writer.print("  * object {}:\n", .{object_i});

                    try writer.writeAll("    * name: ");
                    if (object.p_object_name) |name| {
                        try writer.print("{s}", .{name});
                    } else {
                        try writer.writeAll("null");
                    }
                    try writer.writeByte('\n');

                    try writer.print("    * type: {}\n", .{object.object_type});
                    try writer.print("    * handle: 0x{x}", .{object.object_handle});
                }
            }
        }
    }
}

fn fmtDebugMessage(data: FormatDebugMessageData) std.fmt.Formatter(formatDebugMessage) {
    return .{ .data = data };
}

fn vkDebugCallback(
    severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk.DebugUtilsMessageTypeFlagsEXT,
    data: [*c]const vk.DebugUtilsMessengerCallbackDataEXT,
    user_data: ?*anyopaque,
) callconv(.C) vk.Bool32 {
    var level: std.log.Level = if (severity.error_bit_ext)
        .err
    else if (severity.warning_bit_ext)
        .warn
    else if (severity.info_bit_ext)
        .info
    else if (severity.verbose_bit_ext)
        .debug
    else b: {
        log.err("unknown severity {}", .{severity});
        break :b .err;
    };

    // Ignore or reduce the severity of some AMD warnings
    if (level == .warn) {
        if (data) |d| switch (d.*.message_id_number) {
            // Ignore `BestPractices-vkBindBufferMemory-small-dedicated-allocation` and
            // `BestPractices-vkAllocateMemory-small-allocation`, our whole rendering strategy is
            // designed around this but we often have so little total memory that we trip it anyway!
            280337739, -40745094 => return vk.FALSE,
            // Don't warn us that validation is on every time validation is on, but do log it as
            // debug
            615892639, 2132353751, 1734198062 => level = .debug,
            // Don't warn us about skipping unsupported drivers, but do log it as debug
            0 => if (d.*.p_message_id_name) |name| {
                if (std.mem.eql(u8, std.mem.span(name), "Loader Message")) {
                    level = .debug;
                }
            },
            else => {},
        };
    }

    // Otherwise log them
    const format = "{}";
    const args = .{fmtDebugMessage(.{
        .severity = severity,
        .message_type = message_type,
        .data = data,
        .user_data = user_data,
    })};
    switch (level) {
        .err => {
            log.err(format, args);
            return vk.TRUE;
        },
        .warn => {
            log.warn(format, args);
            return vk.FALSE;
        },
        .info => {
            log.info(format, args);
            return vk.FALSE;
        },
        .debug => {
            log.debug(format, args);
            return vk.FALSE;
        },
    }
}

inline fn setName(
    device: vk.DeviceProxy,
    object: anytype,
    debug_name: Ctx.DebugName,
    validation: bool,
) void {
    if (!validation) return;

    const type_name, const object_type = switch (@TypeOf(object)) {
        vk.Buffer => .{ "Buffer", .buffer },
        vk.CommandBuffer => .{ "Command Buffer", .command_buffer },
        vk.CommandPool => .{ "Command Pool", .command_pool },
        vk.DescriptorPool => .{ "Descriptor Pool", .descriptor_pool },
        vk.DescriptorSet => .{ "Descriptor Set", .descriptor_set },
        vk.DescriptorSetLayout => .{ "Descriptor Set Layout", .descriptor_set_layout },
        vk.DeviceMemory => .{ "Device Memory", .device_memory },
        vk.Fence => .{ "Fence", .fence },
        vk.Framebuffer => .{ "Framebuf", .framebuffer },
        vk.Image => .{ "Image", .image },
        vk.ImageView => .{ "Image View", .image_view },
        vk.Pipeline => .{ "Pipeline", .pipeline },
        vk.PipelineLayout => .{ "Pipeline Layout", .pipeline_layout },
        vk.QueryPool => .{ "Query Pool", .query_pool },
        vk.Queue => .{ "Queue", .queue },
        vk.RenderPass => .{ "Render Pass", .render_pass },
        vk.Sampler => .{ "Sampler", .sampler },
        vk.Semaphore => .{ "Semaphore", .semaphore },
        vk.ShaderModule => .{ "Shader Module", .shader_module },
        vk.SwapchainKHR => .{ "Swapchain", .swapchain_khr },
        else => @compileError("unexpected type: " ++ @typeName(@TypeOf(object))),
    };

    var buf: [64:0]u8 = undefined;
    buf[buf.len] = 0;
    const name: [:0]const u8 = if (debug_name.index) |i| b: {
        break :b std.fmt.bufPrintZ(&buf, "{s} - {s} {}", .{ type_name, debug_name.str, i }) catch |err| switch (err) {
            error.NoSpaceLeft => buf[0..],
        };
    } else b: {
        break :b std.fmt.bufPrintZ(&buf, "{s} - {s}", .{ type_name, debug_name.str }) catch |err| switch (err) {
            error.NoSpaceLeft => buf[0..],
        };
    };

    device.setDebugUtilsObjectNameEXT(&.{
        .object_type = object_type,
        .object_handle = @intFromEnum(object),
        .p_object_name = name,
    }) catch |err| @panic(@errorName(err));
}

/// Assumes a buffer is null terminated, and returns the string it contains. If it turns out not to
/// be null terminated, the whole buffer is returned.
fn bufToStr(buf: anytype) []const u8 {
    comptime assert(@typeInfo(@TypeOf(buf)) == .pointer);
    for (buf, 0..) |c, i| {
        if (c == 0) return buf[0..i];
    }
    return buf;
}
