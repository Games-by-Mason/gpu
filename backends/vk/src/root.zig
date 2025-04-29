const std = @import("std");
const math = std.math;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const vk = @import("vulkan");
const log = std.log.scoped(.gpu);
const gpu = @import("gpu");
const Ctx = gpu.Ctx;
const tracy = gpu.tracy;
const Zone = tracy.Zone;
const TracyQueue = tracy.GpuQueue;
const global_options = gpu.options;

pub const vulkan = @import("vulkan");

const vk_version = vk.makeApiVersion(0, 1, 3, 0);

// Context
surface: vk.SurfaceKHR,
base_wrapper: vk.BaseWrapper,
device: vk.DeviceProxy,
instance: vk.InstanceProxy,
physical_device: PhysicalDevice,
swapchain: Swapchain,
debug_messenger: vk.DebugUtilsMessengerEXT,

// Queues & commands
timestamp_period: f32,
queue: vk.Queue,
queue_family_index: u32,
cmd_pools: [global_options.max_frames_in_flight]vk.CommandPool,

// Synchronization
image_availables: [global_options.max_frames_in_flight]vk.Semaphore,
ready_for_present: [global_options.max_frames_in_flight]vk.Semaphore,
cmd_pool_ready: [global_options.max_frames_in_flight]vk.Fence,

// The current swapchain image index. Other APIs track this automatically, Vulkan appears to allow
// you to actually present them out of order, but we never want to do this and it wouldn't map to
// other APIs, so we track it ourselves here.
image_index: ?u32 = null,

// Tracy info
tracy_query_pools: [global_options.max_frames_in_flight]vk.QueryPool,

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

const graphics_queue_name = "Graphics Queue";

pub fn init(options: Ctx.InitOptionsImpl(InitOptions)) @This() {
    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    log.info("Graphics API: Vulkan {}.{}.{} (variant {})", .{
        vk_version.major,
        vk_version.minor,
        vk_version.patch,
        vk_version.variant,
    });

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

    log.info("Instance Extensions: {s}", .{instance_extensions.items});
    log.info("Layers: {s}", .{layers.items});
    ext_zone.end();

    const inst_handle_zone = tracy.Zone.begin(.{ .name = "create instance handle", .src = @src() });
    const instance_handle = base_wrapper.createInstance(&.{
        .p_application_info = &.{
            .api_version = @bitCast(vk_version),
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
    log.info("Device Extensions: {s}", .{required_device_extensions.items});

    log.info("All Devices:", .{});
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
        const queue_family_index: ?u32 = for (queue_family_properties, 0..) |qfp, qfi| {
            // Check for present
            if (instance_proxy.getPhysicalDeviceSurfaceSupportKHR(
                device,
                @intCast(qfi),
                surface,
            ) catch |err| @panic(@errorName(err)) != vk.TRUE) continue;
            // Check for graphics and compute. We don't check the transfer bit since graphics and
            // compute imply it, and it's not required to be set if they are.
            if (!qfp.queue_flags.graphics_bit or !qfp.queue_flags.compute_bit) continue;
            // Break with the first compatible queue
            break @intCast(qfi);
        } else null;

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

        log.info("  {}. {s}", .{ i, bufToStr(&properties.device_name) });
        log.debug("    * device api version: {}", .{@as(vk.Version, @bitCast(properties.api_version))});
        log.debug("    * device type: {}", .{properties.device_type});
        log.debug("    * queue family index: {?}", .{queue_family_index});
        log.debug("    * present mode: {?}", .{present_mode});
        log.debug("    * surface format: {?}", .{surface_format});
        log.debug("    * features: {}", .{all_features});
        log.debug("    * max indirect draw count: {}", .{properties.limits.max_draw_indirect_count});
        log.debug("    * min uniform buffer offset alignment: {}", .{properties.limits.min_uniform_buffer_offset_alignment});
        log.debug("    * min storage buffer offset alignment: {}", .{properties.limits.min_storage_buffer_offset_alignment});
        if (!extensions_supported) {
            log.debug("    * missing extensions:", .{});
            var key_iterator = missing_device_extensions.keyIterator();
            while (key_iterator.next()) |key| {
                log.debug("      * {s}", .{key.*});
            }
        }

        const composite_alpha: ?vk.CompositeAlphaFlagsKHR = b: {
            if (surface_capabilities) |sc| {
                const supported = sc.supported_composite_alpha;
                log.debug("    * supported composite alpha: {}", .{supported});
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

        const device_version: vk.Version = @bitCast(properties.api_version);
        const version_compatible = device_version.variant == vk_version.variant and
            @as(u32, @bitCast(device_version)) >= @as(u32, @bitCast(vk_version));
        const compatible = version_compatible and queue_family_index != null and extensions_supported and composite_alpha != null and supports_required_features;

        if (compatible) {
            log.debug("    * rank: {}", .{rank});
        } else {
            log.debug("    * rank: incompatible", .{});
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
                .queue_family_index = queue_family_index.?,
            };
        }
    }

    if (best_physical_device.device == .null_handle) {
        @panic("NoSupportedDevices");
    }

    log.info("Best Device: {s} ({})", .{ bufToStr(&best_physical_device.name), best_physical_device.index });
    devices_zone.end();

    // Iterate over the available queues, and find indices for the various queue types requested
    const queue_zone = tracy.Zone.begin(.{ .name = "queue setup", .src = @src() });
    const queue_family_properties = instance_proxy.getPhysicalDeviceQueueFamilyPropertiesAlloc(best_physical_device.device, gpa) catch |err| @panic(@errorName(err));
    defer gpa.free(queue_family_properties);

    const queue_family_allocated: []u8 = gpa.alloc(u8, queue_family_properties.len) catch @panic("OOM");
    defer gpa.free(queue_family_allocated);
    for (queue_family_allocated) |*c| c.* = 0;

    const queue_create_infos: [1]vk.DeviceQueueCreateInfo = .{.{
        .queue_family_index = best_physical_device.queue_family_index,
        .queue_count = 1,
        .p_queue_priorities = &.{1.0},
    }};
    queue_zone.end();

    const device_proxy_zone = tracy.Zone.begin(.{ .name = "create device proxy", .src = @src() });
    var device_features_11: vk.PhysicalDeviceVulkan11Features = .{
        .shader_draw_parameters = vk.TRUE,
    };
    var device_features_12: vk.PhysicalDeviceVulkan12Features = .{
        .host_query_reset = @intFromBool(options.timestamp_queries),
        .p_next = &device_features_11,
    };
    const device_features_13: vk.PhysicalDeviceVulkan13Features = .{
        // Support required by Vulkan 1.3
        .dynamic_rendering = vk.TRUE,
        .p_next = &device_features_12,
    };
    const device_features: vk.PhysicalDeviceFeatures = .{
        .draw_indirect_first_instance = vk.TRUE,
        .multi_draw_indirect = vk.TRUE,
        .sampler_anisotropy = @intFromBool(best_physical_device.sampler_anisotropy),
    };
    const device_create_info: vk.DeviceCreateInfo = .{
        .p_queue_create_infos = &queue_create_infos,
        .queue_create_info_count = @intCast(queue_create_infos.len),
        .p_enabled_features = &device_features,
        .enabled_extension_count = @intCast(required_device_extensions.items.len),
        .pp_enabled_extension_names = required_device_extensions.items.ptr,
        .p_next = &device_features_13,
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

    timestamp_zone.end();

    const swapchain = Swapchain.init(
        instance_proxy,
        options.framebuf_size,
        device,
        best_physical_device,
        surface,
        .null_handle,
        options.validation,
    );

    const command_pools_zone = tracy.Zone.begin(.{ .name = "create command pools", .src = @src() });
    var cmd_pools: [global_options.max_frames_in_flight]vk.CommandPool = undefined;
    for (&cmd_pools, 0..) |*pool, i| {
        pool.* = device.createCommandPool(&.{
            .flags = .{ .transient_bit = true },
            .queue_family_index = best_physical_device.queue_family_index,
        }, null) catch |err| @panic(@errorName(err));
        setName(device, pool.*, .{ .str = "Graphics", .index = i }, options.validation);
    }
    command_pools_zone.end();

    const sync_primitives_zone = tracy.Zone.begin(.{ .name = "create sync primitives", .src = @src() });
    var image_availables: [global_options.max_frames_in_flight]vk.Semaphore = undefined;
    for (0..global_options.max_frames_in_flight) |i| {
        image_availables[i] = device.createSemaphore(&.{}, null) catch |err| @panic(@errorName(err));
        setName(device, image_availables[i], .{ .str = "Image Available", .index = i }, options.validation);
    }

    var ready_for_present: [global_options.max_frames_in_flight]vk.Semaphore = undefined;
    for (&ready_for_present, 0..) |*semaphore, frame| {
        semaphore.* = device.createSemaphore(&.{}, null) catch |err| @panic(@errorName(err));
        setName(device, semaphore.*, .{
            .str = "Ready For Present",
            .index = frame,
        }, options.validation);
    }

    var cmd_pool_ready: [global_options.max_frames_in_flight]vk.Fence = undefined;
    for (&cmd_pool_ready, 0..) |*fence, frame| {
        fence.* = device.createFence(&.{
            .p_next = null,
            .flags = .{ .signaled_bit = true },
        }, null) catch |err| @panic(@errorName((err)));
        setName(device, fence.*, .{
            .str = "Command Pool Fence",
            .index = frame,
        }, options.validation);
    }
    sync_primitives_zone.end();

    var tracy_query_pools: [global_options.max_frames_in_flight]vk.QueryPool = @splat(.null_handle);
    if (tracy.enabled and options.timestamp_queries) {
        const query_pool_zone = tracy.Zone.begin(.{ .name = "create tracy query pools", .src = @src() });
        defer query_pool_zone.end();

        for (&tracy_query_pools, 0..) |*pool, i| {
            pool.* = device.createQueryPool(&.{
                .query_type = .timestamp,
                .query_count = Ctx.Zone.TracyQueryId.cap,
            }, null) catch |err| @panic(@errorName(err));
            setName(device, pool.*, .{ .str = "Tracy", .index = i }, options.validation);
            device.resetQueryPool(pool.*, 0, Ctx.Zone.TracyQueryId.cap);
        }
    }

    const queue = device.getDeviceQueue(best_physical_device.queue_family_index, 0);
    setName(device, queue, .{ .str = graphics_queue_name }, options.validation);

    return .{
        .surface = surface,
        .base_wrapper = base_wrapper,
        .debug_messenger = debug_messenger,
        .instance = instance_proxy,
        .device = device,
        .swapchain = swapchain,
        .cmd_pools = cmd_pools,
        .image_availables = image_availables,
        .ready_for_present = ready_for_present,
        .cmd_pool_ready = cmd_pool_ready,
        .physical_device = best_physical_device,
        .timestamp_period = timestamp_period,
        .queue = queue,
        .queue_family_index = best_physical_device.queue_family_index,
        .tracy_query_pools = tracy_query_pools,
        .timestamp_queries = options.timestamp_queries,
    };
}

pub fn deinit(self: *Ctx, gpa: Allocator) void {
    // Destroy the Tracy data
    for (self.backend.tracy_query_pools) |pool| {
        self.backend.device.destroyQueryPool(pool, null);
    }

    // Destroy internal sync state
    for (self.backend.ready_for_present) |semaphore| {
        self.backend.device.destroySemaphore(semaphore, null);
    }
    for (self.backend.image_availables) |semaphore| {
        self.backend.device.destroySemaphore(semaphore, null);
    }
    for (self.backend.cmd_pool_ready) |fence| {
        self.backend.device.destroyFence(fence, null);
    }

    // Destroy command state
    for (self.backend.cmd_pools) |pool| {
        self.backend.device.destroyCommandPool(pool, null);
    }

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
) Ctx.DedicatedAllocation(Ctx.DedicatedBuf(.{})) {
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
    const dedicated_alloc_info: vk.MemoryDedicatedAllocateInfo = .{
        .buffer = buffer,
    };
    const memory = self.backend.device.allocateMemory(&.{
        .allocation_size = reqs.size,
        .memory_type_index = memory_type_index,
        .p_next = &dedicated_alloc_info,
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
        .dedicated = .{
            .buf = .fromBackendType(buffer),
            .memory = .fromBackendType(memory),
        },
        .size = reqs.size,
    };
}

pub fn dedicatedUploadBufCreate(
    self: *Ctx,
    name: Ctx.DebugName,
    kind: Ctx.BufKind,
    size: u64,
    prefer_device_local: bool,
) Ctx.DedicatedAllocation(Ctx.DedicatedUploadBuf(.{})) {
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
    const dedicated_alloc_info: vk.MemoryDedicatedAllocateInfo = .{
        .buffer = buffer,
    };
    const memory = self.backend.device.allocateMemory(&.{
        .allocation_size = reqs.size,
        .memory_type_index = memory_type_index,
        .p_next = &dedicated_alloc_info,
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
        .dedicated = .{
            .buf = .fromBackendType(buffer),
            .memory = .fromBackendType(memory),
            .data = data,
        },
        .size = reqs.size,
    };
}

pub fn dedicatedReadbackBufCreate(
    self: *Ctx,
    name: Ctx.DebugName,
    kind: Ctx.BufKind,
    size: u64,
) Ctx.DedicatedAllocation(Ctx.DedicatedReadbackBuf(.{})) {
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
    const dedicated_alloc_info: vk.MemoryDedicatedAllocateInfo = .{
        .buffer = buffer,
    };
    const memory = self.backend.device.allocateMemory(&.{
        .allocation_size = reqs.size,
        .memory_type_index = memory_type_index,
        .p_next = &dedicated_alloc_info,
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
        .dedicated = .{
            .buf = .fromBackendType(buffer),
            .memory = .fromBackendType(memory),
            .data = mapping[0..size],
        },
        .size = reqs.size,
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

pub fn zoneBegin(self: *Ctx, options: Ctx.Zone.BeginOptions) Ctx.Zone {
    if (self.backend.debug_messenger != .null_handle) {
        self.backend.device.cmdBeginDebugUtilsLabelEXT(options.cb.asBackendType(), &.{
            .p_label_name = options.loc.name orelse options.loc.function,
            .color = .{
                @as(f32, @floatFromInt(options.loc.color.r)) / 255.0,
                @as(f32, @floatFromInt(options.loc.color.g)) / 255.0,
                @as(f32, @floatFromInt(options.loc.color.b)) / 255.0,
                @as(f32, @floatFromInt(options.loc.color.a)) / 255.0,
            },
        });
    }
    if (tracy.enabled and self.timestamp_queries) {
        const query_id: Ctx.Zone.TracyQueryId = .next(self);
        self.device.tracy_queue.beginZone(.{
            .query_id = @bitCast(query_id),
            .loc = options.loc,
        });
        self.backend.device.cmdWriteTimestamp(
            options.cb.asBackendType(),
            .{ .top_of_pipe_bit = true },
            self.backend.tracy_query_pools[self.frame],
            query_id.index,
        );
    }
    return .{};
}

pub fn zoneEnd(self: *Ctx, cb: Ctx.CmdBuf) void {
    if (self.backend.debug_messenger != .null_handle) {
        self.backend.device.cmdEndDebugUtilsLabelEXT(cb.asBackendType());
    }

    if (tracy.enabled and self.timestamp_queries) {
        const query_id: Ctx.Zone.TracyQueryId = .next(self);
        self.backend.device.cmdWriteTimestamp(
            cb.asBackendType(),
            .{ .bottom_of_pipe_bit = true },
            self.backend.tracy_query_pools[self.frame],
            query_id.index,
        );
        self.device.tracy_queue.endZone(@bitCast(query_id));
    }
}

pub fn combinedCmdBufCreate(
    self: *Ctx,
    options: Ctx.CombinedCmdBufCreateOptions,
) Ctx.CombinedCmdBuf(null) {
    var cbs = [_]vk.CommandBuffer{.null_handle};
    self.backend.device.allocateCommandBuffers(&.{
        .command_pool = self.backend.cmd_pools[self.frame],
        .level = .primary,
        .command_buffer_count = cbs.len,
    }, &cbs) catch |err| @panic(@errorName(err));
    const cb = cbs[0];
    setName(self.backend.device, cb, .{
        .str = options.loc.name orelse options.loc.function,
    }, self.backend.debug_messenger != .null_handle);

    self.backend.device.beginCommandBuffer(cb, &.{
        .flags = .{ .one_time_submit_bit = true },
        .p_inheritance_info = null,
    }) catch |err| @panic(@errorName(err));

    const zone = Ctx.Zone.begin(self, .{
        .cb = .fromBackendType(cb),
        .loc = options.loc,
    });

    if (options.kind == .graphics) {
        self.backend.device.cmdBeginRendering(cb, &.{
            .flags = .{},
            .render_area = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.backend.swapchain.swap_extent,
            },
            .layer_count = 1,
            .view_mask = 0,
            .color_attachment_count = 1,
            .p_color_attachments = &.{.{
                .image_view = options.out.asBackendType(),
                .image_layout = .color_attachment_optimal,
                .resolve_mode = .{},
                .resolve_image_view = .null_handle,
                .resolve_image_layout = .undefined,
                .load_op = switch (options.load_op) {
                    .clear_color => .clear,
                    .load => .load,
                    .dont_care => .dont_care,
                },
                .store_op = .store,
                .clear_value = switch (options.load_op) {
                    .clear_color => |color| .{ .color = .{ .float_32 = color } },
                    else => .{ .color = .{ .float_32 = .{ 0.0, 0.0, 0.0, 0.0 } } },
                },
            }},
            .p_depth_attachment = null,
            .p_stencil_attachment = null,
        });
    }

    return .{
        .cb = .fromBackendType(cb),
        .bindings = options.bindings,
        .zone = zone,
    };
}

pub fn cmdBufGraphicsAppend(
    self: *Ctx,
    combined: Ctx.CombinedCmdBuf(null),
    options: Ctx.CombinedCmdBuf(null).AppendGraphicsCmdsOptions,
) void {
    const cb = combined.cb.asBackendType();

    const zone = Ctx.Zone.begin(self, .{
        .cb = combined.cb,
        .loc = options.loc,
    });
    defer zone.end(self, combined.cb);

    const bindings = combined.bindings;

    for (options.cmds) |draw| {
        if (draw.combined_pipeline.pipeline != bindings.pipeline) {
            bindings.pipeline = draw.combined_pipeline.pipeline;

            // Bind the pipeline
            self.backend.device.cmdBindPipeline(
                cb,
                .graphics,
                draw.combined_pipeline.pipeline.asBackendType(),
            );

            if (!bindings.dynamic_state) {
                bindings.dynamic_state = true;

                self.backend.device.cmdSetViewport(cb, 0, 1, &.{.{
                    .x = 0.0,
                    .y = 0.0,
                    .width = @floatFromInt(self.backend.swapchain.swap_extent.width),
                    .height = @floatFromInt(self.backend.swapchain.swap_extent.height),
                    .min_depth = 0.0,
                    .max_depth = 1.0,
                }});

                self.backend.device.cmdSetScissor(cb, 0, 1, &.{.{
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
                cb,
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
                    cb,
                    indices.asBackendType(),
                    0,
                    .uint16,
                );
            }

            self.backend.device.cmdDrawIndexedIndirect(
                cb,
                draw.args.buf.asBackendType(),
                draw.args.offset,
                draw.args_count,
                @sizeOf(Ctx.DrawCmd.IndexedIndirect),
            );
        } else {
            self.backend.device.cmdDrawIndirect(
                cb,
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
    combined: Ctx.CombinedCmdBuf(null),
    kind: Ctx.CmdBufKind,
) void {
    const cb = combined.cb.asBackendType();

    if (kind == .graphics) {
        self.backend.device.cmdEndRendering(cb);
    }

    combined.zone.end(self, combined.cb);
    self.backend.device.endCommandBuffer(cb) catch |err| @panic(@errorName(err));

    {
        const queue_submit_zone = Zone.begin(.{ .name = "queue submit", .src = @src() });
        defer queue_submit_zone.end();
        const cbs = [_]vk.CommandBuffer{cb};
        const submit_infos = [_]vk.SubmitInfo{.{
            .wait_semaphore_count = 0,
            .p_wait_semaphores = &.{},
            .p_wait_dst_stage_mask = &.{},
            .command_buffer_count = cbs.len,
            .p_command_buffers = &cbs,
            .signal_semaphore_count = 0,
            .p_signal_semaphores = &.{},
            .p_next = null,
        }};
        self.backend.device.queueSubmit(
            self.backend.queue,
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

pub fn acquireNextImage(self: *Ctx) ?Ctx.ImageView {
    // Acquire the image
    const acquire_result = b: {
        const acquire_zone = Zone.begin(.{
            .src = @src(),
            .name = "acquire next image",
        });
        defer acquire_zone.end();
        break :b self.backend.device.acquireNextImageKHR(
            self.backend.swapchain.swapchain,
            std.math.maxInt(u64),
            self.backend.image_availables[self.frame],
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
    assert(self.backend.image_index == null);
    self.backend.image_index = acquire_result.image_index;

    // Transition it to the right format
    {
        const transition_zone = Zone.begin(.{ .name = "prepare swapchain image", .src = @src() });
        defer transition_zone.end();

        var cbs = [_]vk.CommandBuffer{.null_handle};
        self.backend.device.allocateCommandBuffers(&.{
            .command_pool = self.backend.cmd_pools[self.frame],
            .level = .primary,
            .command_buffer_count = cbs.len,
        }, &cbs) catch |err| @panic(@errorName(err));
        const cb = cbs[0];
        setName(self.backend.device, cb, .{
            .str = "Prepare Swapchain Image",
        }, self.backend.debug_messenger != .null_handle);

        {
            self.backend.device.beginCommandBuffer(cb, &.{
                .flags = .{ .one_time_submit_bit = true },
                .p_inheritance_info = null,
            }) catch |err| @panic(@errorName(err));
            defer self.backend.device.endCommandBuffer(cb) catch |err| @panic(@errorName(err));

            const zone = Ctx.Zone.begin(self, .{
                .cb = .fromBackendType(cb),
                .loc = .init(.{ .name = "prepare swapchain image", .src = @src() }),
            });
            defer zone.end(self, .fromBackendType(cb));

            self.backend.device.cmdPipelineBarrier(
                cb,
                .{ .top_of_pipe_bit = true },
                .{ .color_attachment_output_bit = true },
                .{},
                0,
                null,
                0,
                null,
                1,
                &.{.{
                    .src_access_mask = .{},
                    .dst_access_mask = .{ .color_attachment_write_bit = true },
                    .old_layout = .undefined,
                    .new_layout = .color_attachment_optimal,
                    .src_queue_family_index = 0, // Ignored
                    .dst_queue_family_index = 0, // Ignored
                    .image = self.backend.swapchain.images.get(acquire_result.image_index),
                    .subresource_range = .{
                        .aspect_mask = .{ .color_bit = true },
                        .base_mip_level = 0,
                        .level_count = 1,
                        .base_array_layer = 0,
                        .layer_count = 1,
                    },
                }},
            );
        }

        self.backend.device.queueSubmit(
            self.backend.queue,
            1,
            &.{.{
                .wait_semaphore_count = 1,
                .p_wait_semaphores = &.{self.backend.image_availables[self.frame]},
                .p_wait_dst_stage_mask = &.{.{ .top_of_pipe_bit = true }},
                .command_buffer_count = 1,
                .p_command_buffers = &.{cb},
                .signal_semaphore_count = 0,
                .p_signal_semaphores = &.{},
                .p_next = null,
            }},
            .null_handle,
        ) catch |err| @panic(@errorName(err));
    }

    return .fromBackendType(self.backend.swapchain.views.get(acquire_result.image_index));
}

pub fn frameStart(self: *Ctx) void {
    self.backend.image_index = null;

    {
        const wait_zone = Zone.begin(.{
            .src = @src(),
            .name = "wait for cmd pool",
        });
        defer wait_zone.end();
        const cmd_pool_fence = self.backend.cmd_pool_ready[self.frame];
        assert(self.backend.device.waitForFences(
            1,
            &.{cmd_pool_fence},
            vk.TRUE,
            std.math.maxInt(u64),
        ) catch |err| @panic(@errorName(err)) == .success);
        self.backend.device.resetFences(1, &.{cmd_pool_fence}) catch |err| @panic(@errorName(err));
    }

    const reset_cmd_pool_zone = Zone.begin(.{
        .src = @src(),
        .name = "reset cmd pool",
    });
    const cmd_pool = self.backend.cmd_pools[self.frame];
    log.err("try to reset {}", .{self.frame});
    self.backend.device.resetCommandPool(cmd_pool, .{}) catch |err| @panic(@errorName(err));
    log.err("done", .{});
    reset_cmd_pool_zone.end();

    if (tracy.enabled and self.timestamp_queries) {
        const tracy_query_pool_zone = Zone.begin(.{
            .src = @src(),
            .name = "tracy query pool",
        });
        defer tracy_query_pool_zone.end();

        const queries = self.tracy_queries[self.frame];
        if (queries > 0) {
            var results: [Ctx.Zone.TracyQueryId.cap * 2]u64 = undefined;
            const result = self.backend.device.getQueryPoolResults(
                self.backend.tracy_query_pools[self.frame],
                0,
                queries,
                @sizeOf(u64) * @as(u32, queries) * 2,
                &results,
                @sizeOf(u64) * 2,
                .{
                    .@"64_bit" = true,
                    .with_availability_bit = true,
                },
            ) catch |err| @panic(@errorName(err));
            switch (result) {
                // It's possible that the caller generated some command buffers that they decided
                // not to submit, so we might sometimes get not ready here. That's fine--we'll skip
                // those timestamps using the availability bit.
                .success, .not_ready => {},
                else => @panic(@tagName(result)),
            }

            for (0..queries) |i| {
                const available = results[i * 2 + 1] != 0;
                if (available) {
                    const time = results[i * 2];
                    self.device.tracy_queue.emitTime(.{
                        .query_id = @bitCast(Ctx.Zone.TracyQueryId{
                            .index = @intCast(i),
                            .frame = self.frame,
                        }),
                        .gpu_time = time,
                    });
                }
            }

            var cbs = [_]vk.CommandBuffer{.null_handle};
            self.backend.device.allocateCommandBuffers(&.{
                .command_pool = self.backend.cmd_pools[self.frame],
                .level = .primary,
                .command_buffer_count = cbs.len,
            }, &cbs) catch |err| @panic(@errorName(err));
            const cb = cbs[0];
            setName(self.backend.device, cb, .{
                .str = "Clear Tracy Query Pool",
            }, self.backend.debug_messenger != .null_handle);

            {
                self.backend.device.beginCommandBuffer(cb, &.{
                    .flags = .{ .one_time_submit_bit = true },
                    .p_inheritance_info = null,
                }) catch |err| @panic(@errorName(err));
                defer self.backend.device.endCommandBuffer(cb) catch |err| @panic(@errorName(err));
                self.backend.device.cmdResetQueryPool(
                    cb,
                    self.backend.tracy_query_pools[self.frame],
                    0,
                    Ctx.Zone.TracyQueryId.cap,
                );
            }

            self.backend.device.queueSubmit(
                self.backend.queue,
                1,
                &.{.{
                    .wait_semaphore_count = 0,
                    .p_wait_semaphores = &.{},
                    .p_wait_dst_stage_mask = &.{},
                    .command_buffer_count = 1,
                    .p_command_buffers = &.{cb},
                    .signal_semaphore_count = 0,
                    .p_signal_semaphores = &.{},
                    .p_next = null,
                }},
                .null_handle,
            ) catch |err| @panic(@errorName(err));
        }
    }
}

pub fn getDevice(self: *const @This()) Ctx.Device {
    const get_queue_zone = tracy.Zone.begin(.{ .name = "get queues", .src = @src() });
    const calibration = timestampCalibrationImpl(self.device, self.timestamp_queries);
    const tracy_queue = TracyQueue.init(.{
        .gpu_time = calibration.gpu,
        .period = self.timestamp_period,
        .context = 0,
        .flags = .{},
        .type = .vulkan,
        .name = graphics_queue_name,
    });
    get_queue_zone.end();

    return .{
        .kind = self.physical_device.ty,
        .uniform_buf_offset_alignment = self.physical_device.min_uniform_buffer_offset_alignment,
        .storage_buf_offset_alignment = self.physical_device.min_storage_buffer_offset_alignment,
        .timestamp_period = self.timestamp_period,
        .tracy_queue = tracy_queue,
    };
}

fn imageOptionsToVk(options: Ctx.ImageOptions) vk.ImageCreateInfo {
    return .{
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
    };
}

fn allocImage(
    self: *Ctx,
    options: Ctx.ImageOptions,
    image: vk.Image,
    reqs: vk.MemoryRequirements,
) Ctx.ImageResultUntyped {
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

    // Allocate memory for the image
    const dedicated_alloc_info: vk.MemoryDedicatedAllocateInfo = .{
        .image = image,
    };
    const memory = self.backend.device.allocateMemory(&.{
        .allocation_size = reqs.size,
        .memory_type_index = memory_type_index,
        .p_next = &dedicated_alloc_info,
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.device, memory, options.name, self.backend.debug_messenger != .null_handle);

    // Bind the image to the memory
    self.backend.device.bindImageMemory(
        image,
        memory,
        0,
    ) catch |err| @panic(@errorName(err));

    // Return the image and dedicated memory
    return .{
        .image = .fromBackendType(image),
        .dedicated = .{
            .unsized = .fromBackendType(memory),
            .size = reqs.size,
        },
    };
}

fn placeImage(
    self: *Ctx,
    image: vk.Image,
    offset: u64,
    memory: vk.DeviceMemory,
) Ctx.ImageResultUntyped {
    self.backend.device.bindImageMemory(image, memory, offset) catch |err| @panic(@errorName(err));
    return .{
        .image = .fromBackendType(image),
        .dedicated = null,
    };
}

pub fn imageCreate(
    self: *Ctx,
    alloc_options: Ctx.Image(.{}).AllocOptions,
    image_options: Ctx.ImageOptions,
) Ctx.ImageResultUntyped {
    // Create the image
    const image = self.backend.device.createImage(&imageOptionsToVk(image_options), null) catch |err| @panic(@errorName(err));
    setName(self.backend.device, image, image_options.name, self.backend.debug_messenger != .null_handle);

    switch (alloc_options) {
        .auto => |auto| {
            // Get the memory requirements
            var dedicated_reqs: vk.MemoryDedicatedRequirements = .{
                .prefers_dedicated_allocation = vk.FALSE,
                .requires_dedicated_allocation = vk.FALSE,
            };
            var reqs2: vk.MemoryRequirements2 = .{
                .memory_requirements = undefined,
                .p_next = &dedicated_reqs,
            };
            self.backend.device.getImageMemoryRequirements2(&.{ .image = image }, &reqs2);
            const reqs = reqs2.memory_requirements;

            // Check whether this should be a dedicated allocation
            const dedicated = dedicated_reqs.prefers_dedicated_allocation == vk.TRUE or
                dedicated_reqs.requires_dedicated_allocation == vk.TRUE;
            if (dedicated) {
                return allocImage(self, image_options, image, reqs);
            } else {
                auto.offset.* = std.mem.alignForward(u64, auto.offset.*, reqs.alignment);
                const new_offset = auto.offset.* + reqs.size;
                if (new_offset > auto.memory.size) @panic("OOB");
                const result = placeImage(
                    self,
                    image,
                    auto.offset.*,
                    auto.memory.unsized.asBackendType(),
                );
                auto.offset.* = new_offset;
                return result;
            }
        },
        .dedicated => {
            var reqs2: vk.MemoryRequirements2 = .{ .memory_requirements = undefined };
            self.backend.device.getImageMemoryRequirements2(&.{ .image = image }, &reqs2);
            const reqs = reqs2.memory_requirements;
            return allocImage(self, image_options, image, reqs);
        },
        .place => |place| {
            var reqs2: vk.MemoryRequirements2 = .{ .memory_requirements = undefined };
            self.backend.device.getImageMemoryRequirements2(&.{ .image = image }, &reqs2);
            const reqs = reqs2.memory_requirements;
            const offset = std.mem.alignForward(u64, place.offset, reqs.alignment);
            const new_offset = offset + reqs.size;
            if (new_offset > place.memory.size) @panic("OOB");
            return placeImage(self, image, offset, place.memory.unsized.asBackendType());
        },
    }
}

pub fn imageDestroy(self: *Ctx, image: Ctx.Image(.{})) void {
    self.backend.device.destroyImage(image.asBackendType(), null);
}

pub fn imageMemoryRequirements(
    self: *Ctx,
    options: Ctx.ImageOptions,
) Ctx.MemoryRequirements {
    var dedicated_reqs: vk.MemoryDedicatedRequirements = .{
        .prefers_dedicated_allocation = vk.FALSE,
        .requires_dedicated_allocation = vk.FALSE,
    };
    var reqs2: vk.MemoryRequirements2 = .{
        .memory_requirements = undefined,
        .p_next = &dedicated_reqs,
    };
    self.backend.device.getDeviceImageMemoryRequirements(&.{
        .p_create_info = &imageOptionsToVk(options),
        .plane_aspect = .{},
    }, &reqs2);
    const reqs = reqs2.memory_requirements;
    return .{
        .size = reqs.size,
        .alignment = reqs.alignment,
        .dedicated_allocation = if (dedicated_reqs.requires_dedicated_allocation == vk.TRUE)
            .required
        else if (dedicated_reqs.prefers_dedicated_allocation == vk.TRUE)
            .preferred
        else
            .discouraged,
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
) Ctx.MemoryUnsized {
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
            var reqs2: vk.MemoryRequirements2 = .{ .memory_requirements = undefined };
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
                &reqs2,
            );
            const reqs = reqs2.memory_requirements;
            const memory_type_bits: std.bit_set.IntegerBitSet(32) = .{
                .mask = reqs.memory_type_bits,
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
            var reqs2: vk.MemoryRequirements2 = .{ .memory_requirements = undefined };
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
                &reqs2,
            );
            const reqs = reqs2.memory_requirements;
            const memory_type_bits: std.bit_set.IntegerBitSet(32) = .{
                .mask = reqs.memory_type_bits,
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

pub fn memoryDestroy(self: *Ctx, memory: Ctx.MemoryUnsized) void {
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
    const rendering_create_info: vk.PipelineRenderingCreateInfo = .{
        .view_mask = 0,
        .color_attachment_count = 1,
        .p_color_attachment_formats = &.{self.backend.physical_device.surface_format.format},
        .depth_attachment_format = .undefined,
        .stencil_attachment_format = .undefined,
    };
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
            .render_pass = .null_handle,
            .subpass = 0,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = -1,
            .p_next = &rendering_create_info,
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

pub fn present(self: *Ctx) u64 {
    {
        const transition_zone = Zone.begin(.{ .name = "finalize swapchain image", .src = @src() });
        defer transition_zone.end();

        var cbs = [_]vk.CommandBuffer{.null_handle};
        self.backend.device.allocateCommandBuffers(&.{
            .command_pool = self.backend.cmd_pools[self.frame],
            .level = .primary,
            .command_buffer_count = cbs.len,
        }, &cbs) catch |err| @panic(@errorName(err));
        const cb = cbs[0];
        setName(self.backend.device, cb, .{
            .str = "Finalize Swapchain Image",
        }, self.backend.debug_messenger != .null_handle);

        {
            const submit_zone = Zone.begin(.{ .name = "prepare", .src = @src() });
            defer submit_zone.end();

            self.backend.device.beginCommandBuffer(cb, &.{
                .flags = .{ .one_time_submit_bit = true },
                .p_inheritance_info = null,
            }) catch |err| @panic(@errorName(err));
            defer self.backend.device.endCommandBuffer(cb) catch |err| @panic(@errorName(err));

            const zone = Ctx.Zone.begin(self, .{
                .cb = .fromBackendType(cb),
                .loc = .init(.{ .name = "finalize swapchain image", .src = @src() }),
            });
            defer zone.end(self, .fromBackendType(cb));

            self.backend.device.cmdPipelineBarrier(
                cb,
                .{ .color_attachment_output_bit = true },
                .{ .bottom_of_pipe_bit = true },
                .{},
                0,
                null,
                0,
                null,
                1,
                &.{.{
                    .src_access_mask = .{ .color_attachment_write_bit = true },
                    .dst_access_mask = .{},
                    .old_layout = .color_attachment_optimal,
                    .new_layout = .present_src_khr,
                    .src_queue_family_index = 0, // Ignored
                    .dst_queue_family_index = 0, // Ignored
                    .image = self.backend.swapchain.images.get(self.backend.image_index.?),
                    .subresource_range = .{
                        .aspect_mask = .{ .color_bit = true },
                        .base_mip_level = 0,
                        .level_count = 1,
                        .base_array_layer = 0,
                        .layer_count = 1,
                    },
                }},
            );
        }

        {
            self.backend.device.queueSubmit(
                self.backend.queue,
                1,
                &.{.{
                    .wait_semaphore_count = 0,
                    .p_wait_semaphores = &.{},
                    .p_wait_dst_stage_mask = &.{},
                    .command_buffer_count = 1,
                    .p_command_buffers = &.{cb},
                    .signal_semaphore_count = 1,
                    .p_signal_semaphores = &.{
                        self.backend.ready_for_present[self.frame],
                    },
                    .p_next = null,
                }},
                self.backend.cmd_pool_ready[self.frame],
            ) catch |err| @panic(@errorName(err));
        }
    }

    const suboptimal_out_of_date, const present_ns = b: {
        const queue_present_zone = Zone.begin(.{ .name = "queue present", .src = @src() });
        defer queue_present_zone.end();
        const swapchain = [_]vk.SwapchainKHR{self.backend.swapchain.swapchain};
        const image_index = [_]u32{self.backend.image_index.?};

        var present_timer = std.time.Timer.start() catch |err| @panic(@errorName(err));
        {
            const blocking_zone = Zone.begin(.{
                .src = @src(),
                .name = "blocking: queue present",
                .color = global_options.blocking_zone_color,
            });
            defer blocking_zone.end();

            const suboptimal_out_of_date = if (self.backend.device.queuePresentKHR(
                self.backend.queue,
                &.{
                    .wait_semaphore_count = 1,
                    .p_wait_semaphores = &.{self.backend.ready_for_present[self.frame]},
                    .swapchain_count = swapchain.len,
                    .p_swapchains = &swapchain,
                    .p_image_indices = &image_index,
                    .p_results = null,
                },
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
    combined: Ctx.CombinedCmdBuf(null),
    comptime max_regions: u32,
    options: Ctx.CombinedCmdBuf(null).AppendTransferCmdsOptions,
) void {
    const cb = combined.cb.asBackendType();

    const zone = Ctx.Zone.begin(self, .{
        .cb = .fromBackendType(cb),
        .loc = options.loc,
    });

    for (options.cmds) |cmd| {
        switch (cmd) {
            .copy_buffer_to_color_image => |cmd_options| {
                const subresource_range: vk.ImageSubresourceRange = .{
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
                };

                // Transition the image to transfer dst optimal
                self.backend.device.cmdPipelineBarrier(
                    cb,
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
                        .subresource_range = subresource_range,
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
                    cb,
                    cmd_options.buf.asBackendType(),
                    cmd_options.image.asBackendType(),
                    .transfer_dst_optimal,
                    @intCast(regions.len),
                    &regions.buffer,
                );

                // Transition to the destination layout
                self.backend.device.cmdPipelineBarrier(
                    cb,
                    .{ .transfer_bit = true },
                    .{ .fragment_shader_bit = true },
                    .{},
                    0,
                    null,
                    0,
                    null,
                    1,
                    &.{.{
                        .src_access_mask = .{ .transfer_write_bit = true },
                        .dst_access_mask = .{ .shader_read_bit = true },
                        .old_layout = .transfer_dst_optimal,
                        .new_layout = layoutToVk(cmd_options.new_layout),
                        .src_queue_family_index = 0, // Ignored
                        .dst_queue_family_index = 0, // Ignored
                        .image = cmd_options.image.asBackendType(),
                        .subresource_range = subresource_range,
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
                    cb,
                    cmd_options.src.asBackendType(),
                    cmd_options.dst.asBackendType(),
                    @intCast(regions.len),
                    &regions.buffer,
                );
            },
        }
    }

    zone.end(self, .fromBackendType(cb));
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
    swap_extent: vk.Extent2D,
    external_framebuf_size: struct { u32, u32 },

    pub fn init(
        instance: vk.InstanceProxy,
        framebuf_size: struct { u32, u32 },
        device: vk.DeviceProxy,
        physical_device: PhysicalDevice,
        surface: vk.SurfaceKHR,
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

        return .{
            .swapchain = swapchain,
            .images = images,
            .views = views,
            .swap_extent = swap_extent,
            .external_framebuf_size = framebuf_size,
        };
    }

    pub fn destroyEverythingExceptSwapchain(self: *@This(), device: vk.DeviceProxy) void {
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
            retired,
            gx.backend.debug_messenger != .null_handle,
        );
        gx.backend.device.destroySwapchainKHR(retired, null);
    }
};

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
    queue_family_index: u32 = undefined,
};

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
        vk.Image => .{ "Image", .image },
        vk.ImageView => .{ "Image View", .image_view },
        vk.Pipeline => .{ "Pipeline", .pipeline },
        vk.PipelineLayout => .{ "Pipeline Layout", .pipeline_layout },
        vk.QueryPool => .{ "Query Pool", .query_pool },
        vk.Queue => .{ "Queue", .queue },
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
