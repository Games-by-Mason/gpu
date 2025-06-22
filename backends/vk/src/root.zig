const std = @import("std");
const builtin = @import("builtin");
const math = std.math;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const btypes = gpu.btypes;
const log = std.log.scoped(.gpu);
const gpu = @import("gpu");
const Gx = gpu.Gx;
const tracy = gpu.tracy;
const Zone = tracy.Zone;
const TracyQueue = tracy.GpuQueue;
const global_options = gpu.global_options;

pub const vk = @import("vulkan");

const vk_version = vk.makeApiVersion(0, 1, 3, 0);

// In practice, there should typically be only two or three.
const max_swapchain_depth = 8;

// Context
surface: vk.SurfaceKHR,
base_wrapper: vk.BaseWrapper,
device: vk.DeviceProxy,
instance: vk.InstanceProxy,
physical_device: PhysicalDevice,
swapchain: Swapchain,
debug_messenger: vk.DebugUtilsMessengerEXT,
pipeline_cache: vk.PipelineCache,

// Queues & commands
timestamp_period: f32,
queue: vk.Queue,
queue_family_index: u32,
cmd_pools: [global_options.max_frames_in_flight]vk.CommandPool,

// Synchronization
image_availables: [global_options.max_frames_in_flight]vk.Semaphore,
ready_for_present: [max_swapchain_depth]vk.Semaphore,
cmd_pool_ready: [global_options.max_frames_in_flight]vk.Fence,

// The current swapchain image index. Other APIs track this automatically, Vulkan appears to allow
// you to actually present them out of order, but we never want to do this and it wouldn't map to
// other APIs, so we track it ourselves here.
image_index: ?u32 = null,

// Tracy info
tracy_query_pools: [global_options.max_frames_in_flight]vk.QueryPool,

pub const Options = struct {
    const CreateSurfaceError = error{
        OutOfHostMemory,
        OutOfDeviceMemory,
        NativeWindowInUseKHR,
        Unknown,
    };

    instance_extensions: []const [*:0]const u8,
    getInstanceProcAddress: vk.PfnGetInstanceProcAddr,
    surface_context: ?*anyopaque,
    createSurface: *const fn (
        instance: vk.Instance,
        context: ?*anyopaque,
        allocation_callbacks: ?*const vk.AllocationCallbacks,
    ) vk.SurfaceKHR,
    /// Allows you to blacklist problematic layers by setting the `VK_LAYERS_DISABLE` environment
    /// variable at runtime.
    ///
    /// If you just want to disable all implicit layers, see also `safe_mode` in the top level
    /// initialization options.
    ///
    /// Syntax matches the environment variable:
    /// * https://vulkan.lunarg.com/doc/view/1.3.236.0/linux/LoaderDebugging.html#user-content-disable-layers
    layers_disable: OsStr.Optional = .none,
};

const graphics_queue_name = "Graphics Queue";

const DeviceFeatures = struct {
    // Keep in sync with `supersetOf`
    vk11: vk.PhysicalDeviceVulkan11Features = .{},
    vk12: vk.PhysicalDeviceVulkan12Features = .{},
    vk13: vk.PhysicalDeviceVulkan13Features = .{},
    vk10: vk.PhysicalDeviceFeatures2 = .{ .features = .{} },

    fn initEmpty(self: *@This()) void {
        self.vk10 = .{ .p_next = &self.vk11, .features = .{} };
        self.vk11 = .{ .p_next = &self.vk12 };
        self.vk12 = .{ .p_next = &self.vk13 };
        self.vk13 = .{};
    }

    const InitRequiredOptions = struct {
        host_query_reset: bool,
        sampler_anisotropy: bool,
    };

    fn initRequired(self: *@This(), options: InitRequiredOptions) void {
        self.initEmpty();

        // 100% of Windows and Linux devices in `vulkan.gpuinfo.org` support these features at the
        // time of writing.
        self.vk12.host_query_reset = @intFromBool(options.host_query_reset);
        self.vk13.synchronization_2 = vk.TRUE;
        self.vk13.dynamic_rendering = vk.TRUE;
        self.vk13.pipeline_creation_cache_control = vk.TRUE;

        // Roadmap 2022
        self.vk10.features.sampler_anisotropy = @intFromBool(options.sampler_anisotropy);
        self.vk12.scalar_block_layout = vk.TRUE;
        self.vk12.runtime_descriptor_array = vk.TRUE;
        self.vk12.descriptor_binding_partially_bound = vk.TRUE;
        self.vk12.shader_uniform_buffer_array_non_uniform_indexing = vk.TRUE;
        self.vk12.shader_sampled_image_array_non_uniform_indexing = vk.TRUE;
        self.vk12.shader_storage_buffer_array_non_uniform_indexing = vk.TRUE;
        self.vk12.shader_storage_image_array_non_uniform_indexing = vk.TRUE;
        self.vk12.shader_uniform_texel_buffer_array_non_uniform_indexing = vk.TRUE;
        self.vk12.shader_storage_texel_buffer_array_non_uniform_indexing = vk.TRUE;

        // Roadmap 2024
        self.vk11.shader_draw_parameters = vk.TRUE;
    }

    fn supersetOf(superset: *@This(), subset: *@This()) bool {
        if (!featuresSupersetOf(superset.vk10.features, subset.vk10.features)) return false;
        if (!featuresSupersetOf(superset.vk11, subset.vk11)) return false;
        if (!featuresSupersetOf(superset.vk12, subset.vk12)) return false;
        if (!featuresSupersetOf(superset.vk13, subset.vk13)) return false;
        return true;
    }

    fn featuresSupersetOf(superset: anytype, subset: @TypeOf(superset)) bool {
        var result = true;
        inline for (std.meta.fields(@TypeOf(superset))) |field| {
            if (field.type != vk.Bool32) {
                comptime assert(std.mem.eql(u8, field.name, "p_next") or
                    std.mem.eql(u8, field.name, "s_type"));
                continue;
            }
            const super_enabled = @field(superset, field.name) == vk.TRUE;
            const sub_enabled = @field(subset, field.name) == vk.TRUE;
            if (sub_enabled and !super_enabled) {
                log.debug("\t* missing feature: {s}", .{field.name});
                result = false;
            }
        }
        return result;
    }

    fn root(self: *@This()) *vk.PhysicalDeviceFeatures2 {
        return &self.vk10;
    }
};

pub fn init(gpa: Allocator, options: Gx.Options) btypes.BackendInitResult {
    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    log.info("Graphics API: Vulkan {}.{}.{} (variant {})", .{
        vk_version.major,
        vk_version.minor,
        vk_version.patch,
        vk_version.variant,
    });

    if (options.safe_mode) {
        log.info("Safe Mode: {}", .{options.safe_mode});
        setenv(.fromLit("VK_LOADER_LAYERS_DISABLE"), .fromLit("~implicit~"));
    }

    if (options.backend.layers_disable.unwrap()) |str| {
        setenv(.fromLit("VK_LOADER_LAYERS_DISABLE"), str);
    }

    // Load the base dispatch function pointers
    const fp_zone = tracy.Zone.begin(.{ .name = "load fps", .src = @src() });
    const getInstProcAddr = options.backend.getInstanceProcAddress;
    const base_wrapper = vk.BaseWrapper.load(getInstProcAddr);
    fp_zone.end();

    // Determine the required layers and extensions
    const ext_zone = tracy.Zone.begin(.{ .name = "layers & extensions", .src = @src() });
    var layers: std.ArrayListUnmanaged([*:0]const u8) = .{};
    defer layers.deinit(gpa);

    var instance_exts: std.ArrayListUnmanaged([*:0]const u8) = .{};
    defer instance_exts.deinit(gpa);
    instance_exts.appendSlice(gpa, options.backend.instance_extensions) catch @panic("OOM");

    const dbg_messenger_info: vk.DebugUtilsMessengerCreateInfoEXT = .{
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

    // Mutable because this copy may be part of a p_next chain
    var instance_dbg_messenger_info = dbg_messenger_info;

    var instance_validation_features: vk.ValidationFeaturesEXT = .{
        .enabled_validation_feature_count = enabled_validation_features.len,
        .p_enabled_validation_features = &enabled_validation_features,
    };

    var create_instance_chain: ?*vk.BaseInStructure = null;

    // Set requested layers, and log all in case any are implicit and end up causing problems
    {
        const val_layer_name = "VK_LAYER_KHRONOS_validation";
        const supported_layers = base_wrapper.enumerateInstanceLayerPropertiesAlloc(gpa) catch |err| @panic(@errorName(err));
        var validation_layer_missing = options.debug.gte(.validate);
        defer gpa.free(supported_layers);

        log.debug("Supported Layers:", .{});
        var val_props: ?vk.LayerProperties = null;
        for (supported_layers) |props| {
            const curr_name = std.mem.span(@as([*:0]const u8, @ptrCast(&props.layer_name)));
            const version: vk.Version = @bitCast(props.spec_version);
            log.debug("  {s} v{}.{}.{} (variant {}, impl {})", .{
                curr_name,
                version.major,
                version.minor,
                version.patch,
                version.variant,
                props.implementation_version,
            });

            if (options.debug.gte(.validate) and std.mem.eql(u8, curr_name, val_layer_name)) {
                appendNext(&create_instance_chain, @ptrCast(&instance_validation_features));
                layers.append(gpa, val_layer_name) catch @panic("OOM");
                validation_layer_missing = false;
                val_props = props;
            }
        }
        if (val_props) |vp| {
            const version: vk.Version = @bitCast(vp.spec_version);
            log.info("{s} v{}.{}.{} (variant {}, impl {})", .{
                val_layer_name,
                version.major,
                version.minor,
                version.patch,
                version.variant,
                vp.implementation_version,
            });
        }
        if (validation_layer_missing) {
            log.warn("{s}: requested but not found", .{val_layer_name});
        }
    }

    // Try to enable the debug extension
    const debug = if (options.debug.gte(.output)) b: {
        const dbg_ext_name = vk.extensions.ext_debug_utils.name;
        const supported_instance_exts = base_wrapper.enumerateInstanceExtensionPropertiesAlloc(
            null,
            gpa,
        ) catch |err| @panic(@errorName(err));
        defer gpa.free(supported_instance_exts);
        for (supported_instance_exts) |props| {
            const curr_name = std.mem.span(@as([*:0]const u8, @ptrCast(&props.extension_name)));
            if (std.mem.eql(u8, dbg_ext_name, curr_name)) {
                log.info("{s} v{}", .{ dbg_ext_name, props.spec_version });
                instance_exts.append(gpa, vk.extensions.ext_debug_utils.name) catch @panic("OOM");
                appendNext(&create_instance_chain, @ptrCast(&instance_dbg_messenger_info));
                break :b true;
            }
        } else {
            log.warn("{s}: requested but not found", .{dbg_ext_name});
            break :b false;
        }
    } else false;

    log.debug("Required Instance Extensions: {s}", .{instance_exts.items});
    log.debug("Required Layers: {s}", .{layers.items});
    ext_zone.end();

    const inst_handle_zone = tracy.Zone.begin(.{ .name = "create instance handle", .src = @src() });
    const instance_handle = base_wrapper.createInstance(&.{
        .p_application_info = &.{
            .api_version = @bitCast(vk_version),
            .p_application_name = if (options.app_name) |n| n.ptr else null,
            .application_version = @bitCast(vk.makeApiVersion(
                0,
                options.app_version.major,
                options.app_version.minor,
                options.app_version.patch,
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
        .enabled_extension_count = math.cast(u32, instance_exts.items.len) orelse @panic("overflow"),
        .pp_enabled_extension_names = instance_exts.items.ptr,
        .p_next = create_instance_chain,
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
    const debug_messenger = if (debug) instance_proxy.createDebugUtilsMessengerEXT(
        &dbg_messenger_info,
        null,
    ) catch |err| @panic(@errorName(err)) else .null_handle;
    debug_messenger_zone.end();

    const surface_zone = tracy.Zone.begin(.{ .name = "create surface", .src = @src() });
    const surface = options.backend.createSurface(
        instance_proxy.handle,
        options.backend.surface_context,
        null,
    );
    if (surface == .null_handle) {
        @panic("create surface failed");
    }
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
    log.debug("Required Device Extensions: {s}", .{required_device_extensions.items});

    log.info("All Devices:", .{});
    for (physical_devices, 0..) |device, i| {
        const properties = instance_proxy.getPhysicalDeviceProperties(device);
        log.info("  {}. {s}", .{ i, bufToStr(&properties.device_name) });
        log.debug("\t* device api version: {}", .{@as(vk.Version, @bitCast(properties.api_version))});
        log.debug("\t* device type: {}", .{properties.device_type});
        log.debug("\t* min uniform buffer offset alignment: {}", .{properties.limits.min_uniform_buffer_offset_alignment});
        log.debug("\t* min storage buffer offset alignment: {}", .{properties.limits.min_storage_buffer_offset_alignment});

        var features: DeviceFeatures = .{};
        features.initEmpty();
        instance_proxy.getPhysicalDeviceFeatures2(device, features.root());
        var required_features: DeviceFeatures = .{};
        required_features.initRequired(.{
            .host_query_reset = options.timestamp_queries,
            .sampler_anisotropy = false,
        });

        const supports_required_features = features.supersetOf(&required_features);

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
        log.debug("\t* queue family index: {?}", .{queue_family_index});

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
                // Regardless of our surface format, the output color space should be srgb.
                if (surface_format.color_space == .srgb_nonlinear_khr) {
                    // We require at least three channels of whichever color space is requested
                    switch (options.surface_format) {
                        .unorm4x8 => switch (surface_format.format) {
                            // 100% of Windows devices on vulkan.gpuinfo.org support this format and
                            // color space.
                            .b8g8r8a8_unorm => format_rank += 3,
                            // Some fallbacks since support on Linux is more varied, at least
                            // according to the database. I suspect that in practice any machine
                            // capable of running games won't need these fallbacks.
                            .r8g8b8a8_unorm => format_rank += 2,
                            .r8g8b8_unorm,
                            .b8g8r8_unorm,
                            .a8b8g8r8_unorm_pack32,
                            => format_rank += 1,
                            else => {},
                        },
                        .srgb4x8 => switch (surface_format.format) {
                            // 99.89% of Windows devices on vulkan.gpuinfo.org support this format
                            // and color space.
                            .b8g8r8a8_srgb => format_rank += 3,
                            // These should cover the remaining devices. I doubt hardware capable of
                            // running games exists that doesn't support at least one SRGB surface
                            // format, if it does then you need to fall back to a linear swapchain
                            // format and do the conversion yourselves.
                            .r8g8b8a8_srgb => format_rank += 2,
                            .r8g8b8_srgb,
                            .b8g8r8_srgb,
                            .a8b8g8r8_srgb_pack32,
                            => format_rank += 1,
                            else => {},
                        },
                    }
                }

                if (format_rank > best_surface_format_rank) {
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

        log.debug("\t* present mode: {?}", .{present_mode});
        log.debug("\t* surface format: {?}", .{surface_format});
        if (!extensions_supported) {
            log.debug("\t* missing extensions:", .{});
            var key_iterator = missing_device_extensions.keyIterator();
            while (key_iterator.next()) |key| {
                log.debug("  \t* {s}", .{key.*});
            }
        }

        const composite_alpha: ?vk.CompositeAlphaFlagsKHR = b: {
            if (surface_capabilities) |sc| {
                const supported = sc.supported_composite_alpha;
                log.debug("\t* supported composite alpha: {}", .{supported});
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
            log.debug("\t* rank: {}", .{rank});
        } else {
            log.debug("\t* rank: incompatible", .{});
        }

        if (compatible and rank > best_physical_device.rank) {
            const new: PhysicalDevice = .{
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
                .min_texel_buffer_offset_alignment = @intCast(properties.limits.min_texel_buffer_offset_alignment),
                .sampler_anisotropy = features.vk10.features.sampler_anisotropy == vk.TRUE,
                .max_sampler_anisotropy = properties.limits.max_sampler_anisotropy,
                .queue_family_index = queue_family_index.?,
            };
            // Separate line to work around partial assignment on continue
            best_physical_device = new;
        }
    }

    if (best_physical_device.device == .null_handle) {
        @panic("no supported devices");
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
    var features: DeviceFeatures = .{};
    features.initRequired(.{
        .host_query_reset = options.timestamp_queries,
        .sampler_anisotropy = best_physical_device.sampler_anisotropy,
    });
    const device_create_info: vk.DeviceCreateInfo = .{
        .p_queue_create_infos = &queue_create_infos,
        .queue_create_info_count = @intCast(queue_create_infos.len),
        .p_enabled_features = null,
        .enabled_extension_count = @intCast(required_device_extensions.items.len),
        .pp_enabled_extension_names = required_device_extensions.items.ptr,
        .p_next = features.root(),
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

    const command_pools_zone = tracy.Zone.begin(.{ .name = "create command pools", .src = @src() });
    var cmd_pools: [global_options.max_frames_in_flight]vk.CommandPool = undefined;
    for (&cmd_pools, 0..) |*pool, i| {
        pool.* = device.createCommandPool(&.{
            .flags = .{ .transient_bit = true },
            .queue_family_index = best_physical_device.queue_family_index,
        }, null) catch |err| @panic(@errorName(err));
        setName(debug_messenger, device, pool.*, .{ .str = "Graphics", .index = i });
    }
    command_pools_zone.end();

    const sync_primitives_zone = tracy.Zone.begin(.{ .name = "create sync primitives", .src = @src() });
    var image_availables: [global_options.max_frames_in_flight]vk.Semaphore = undefined;
    for (0..global_options.max_frames_in_flight) |i| {
        image_availables[i] = device.createSemaphore(&.{}, null) catch |err| @panic(@errorName(err));
        setName(debug_messenger, device, image_availables[i], .{ .str = "Image Available", .index = i });
    }

    var ready_for_present: [max_swapchain_depth]vk.Semaphore = undefined;
    for (&ready_for_present, 0..) |*semaphore, frame| {
        semaphore.* = device.createSemaphore(&.{}, null) catch |err| @panic(@errorName(err));
        setName(debug_messenger, device, semaphore.*, .{
            .str = "Ready For Present",
            .index = frame,
        });
    }

    var cmd_pool_ready: [global_options.max_frames_in_flight]vk.Fence = undefined;
    for (&cmd_pool_ready, 0..) |*fence, frame| {
        fence.* = device.createFence(&.{
            .p_next = null,
            .flags = .{ .signaled_bit = true },
        }, null) catch |err| @panic(@errorName((err)));
        setName(debug_messenger, device, fence.*, .{
            .str = "Command Pool Fence",
            .index = frame,
        });
    }
    sync_primitives_zone.end();

    var tracy_query_pools: [global_options.max_frames_in_flight]vk.QueryPool = @splat(.null_handle);
    if (tracy.enabled and options.timestamp_queries) {
        const query_pool_zone = tracy.Zone.begin(.{ .name = "create tracy query pools", .src = @src() });
        defer query_pool_zone.end();

        for (&tracy_query_pools, 0..) |*pool, i| {
            pool.* = device.createQueryPool(&.{
                .query_type = .timestamp,
                .query_count = gpu.CmdBuf.TracyQueryId.cap,
            }, null) catch |err| @panic(@errorName(err));
            setName(debug_messenger, device, pool.*, .{ .str = "Tracy", .index = i });
            device.resetQueryPool(pool.*, 0, gpu.CmdBuf.TracyQueryId.cap);
        }
    }

    const queue = device.getDeviceQueue(best_physical_device.queue_family_index, 0);
    setName(debug_messenger, device, queue, .{ .str = graphics_queue_name });

    const pipeline_cache = device.createPipelineCache(&.{
        .flags = .{ .externally_synchronized_bit = true },
        .initial_data_size = 0,
        .p_initial_data = null,
    }, null) catch |err| @panic(@errorName(err));

    const calibration: TimestampCalibration = .init(device, options.timestamp_queries);
    const tracy_queue = TracyQueue.init(.{
        .gpu_time = calibration.gpu,
        .period = timestamp_period,
        .context = 0,
        .flags = .{},
        .type = .vulkan,
        .name = graphics_queue_name,
    });

    return .{
        .backend = .{
            .surface = surface,
            .base_wrapper = base_wrapper,
            .debug_messenger = debug_messenger,
            .pipeline_cache = pipeline_cache,
            .instance = instance_proxy,
            .device = device,
            .swapchain = .empty,
            .cmd_pools = cmd_pools,
            .image_availables = image_availables,
            .ready_for_present = ready_for_present,
            .cmd_pool_ready = cmd_pool_ready,
            .physical_device = best_physical_device,
            .timestamp_period = timestamp_period,
            .queue = queue,
            .queue_family_index = best_physical_device.queue_family_index,
            .tracy_query_pools = tracy_query_pools,
        },
        .device = .{
            .kind = best_physical_device.ty,
            .uniform_buf_offset_alignment = best_physical_device.min_uniform_buffer_offset_alignment,
            .storage_buf_offset_alignment = best_physical_device.min_storage_buffer_offset_alignment,
            .texel_buffer_offset_alignment = best_physical_device.min_texel_buffer_offset_alignment,
            .timestamp_period = timestamp_period,
            .tracy_queue = tracy_queue,
            .surface_format = .fromBackendType(best_physical_device.surface_format.format),
        },
    };
}

pub fn deinit(self: *Gx, gpa: Allocator) void {
    // Destroy the pipeline cache
    self.backend.device.destroyPipelineCache(self.backend.pipeline_cache, null);

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

pub fn bufCreate(
    self: *Gx,
    name: gpu.DebugName,
    kind: gpu.BufKind,
    size: u64,
) gpu.Buf(.{}) {
    // Create the buffer
    const usage_flags = bufUsageFlagsFromKind(kind);
    const buffer = self.backend.device.createBuffer(&.{
        .size = size,
        .usage = usage_flags,
        .sharing_mode = .exclusive,
        .flags = .{},
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.debug_messenger, self.backend.device, buffer, name);

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
    setName(self.backend.debug_messenger, self.backend.device, memory, name);

    // Bind the buffer to the memory
    self.backend.device.bindBufferMemory(
        buffer,
        memory,
        0,
    ) catch |err| @panic(@errorName(err));

    // Return the dedicated buffer
    return .{
        .handle = .fromBackendType(buffer),
        .memory = .fromBackendType(memory),
        .size = reqs.size,
    };
}

pub fn uploadBufCreate(
    self: *Gx,
    name: gpu.DebugName,
    kind: gpu.BufKind,
    size: u64,
    prefer_device_local: bool,
) gpu.UploadBuf(.{}) {
    // Create the buffer
    const usage = bufUsageFlagsFromKind(kind);
    const buffer = self.backend.device.createBuffer(&.{
        .size = size,
        .usage = usage,
        .sharing_mode = .exclusive,
        .flags = .{},
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.debug_messenger, self.backend.device, buffer, name);

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
    setName(self.backend.debug_messenger, self.backend.device, memory, name);

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
        .handle = .fromBackendType(buffer),
        .memory = .fromBackendType(memory),
        .data = data,
    };
}

pub fn readbackBufCreate(
    self: *Gx,
    name: gpu.DebugName,
    kind: gpu.BufKind,
    size: u64,
) gpu.ReadbackBuf(.{}) {
    // Create the buffer
    const buffer = self.backend.device.createBuffer(&.{
        .size = size,
        .usage = bufUsageFlagsFromKind(kind),
        .sharing_mode = .exclusive,
        .flags = .{},
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.debug_messenger, self.backend.device, buffer, name);

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
    setName(self.backend.debug_messenger, self.backend.device, memory, name);

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

    // Return the buffer
    return .{
        .handle = .fromBackendType(buffer),
        .memory = .fromBackendType(memory),
        .data = mapping[0..size],
    };
}

fn bufUsageFlagsFromKind(kind: gpu.BufKind) vk.BufferUsageFlags {
    const result: vk.BufferUsageFlags = .{
        .transfer_src_bit = kind.transfer_src,
        .transfer_dst_bit = kind.transfer_dst,
        .uniform_texel_buffer_bit = kind.uniform_texel,
        .storage_texel_buffer_bit = kind.storage_texel,
        .uniform_buffer_bit = kind.uniform,
        .storage_buffer_bit = kind.storage,
        .index_buffer_bit = kind.index,
        .indirect_buffer_bit = kind.indirect,
    };
    assert(@as(u32, @bitCast(result)) != 0);
    return result;
}

pub fn bufDestroy(self: *Gx, buffer: gpu.BufHandle(.{})) void {
    self.backend.device.destroyBuffer(buffer.asBackendType(), null);
}

pub fn pipelineLayoutCreate(
    self: *Gx,
    options: gpu.Pipeline.Layout.Options,
) gpu.Pipeline.Layout {
    // Create the descriptor set layout
    var descs: std.BoundedArray(vk.DescriptorSetLayoutBinding, global_options.combined_pipeline_layout_create_buf_len) = .{};
    var flags: std.BoundedArray(vk.DescriptorBindingFlags, global_options.combined_pipeline_layout_create_buf_len) = .{};
    for (options.descs, 0..) |desc, i| {
        descs.append(.{
            .binding = @intCast(i),
            .descriptor_type = switch (desc.kind) {
                .sampler => .sampler,
                .combined_image_sampler => .combined_image_sampler,
                .sampled_image => .sampled_image,
                .storage_image => .storage_image,
                .uniform_buffer => .uniform_buffer,
                .storage_buffer => .storage_buffer,
            },
            .descriptor_count = desc.count,
            .stage_flags = .{
                .vertex_bit = desc.stages.vertex,
                .fragment_bit = desc.stages.fragment,
                .compute_bit = desc.stages.compute,
            },
            .p_immutable_samplers = null,
        }) catch @panic("OOB");
        flags.appendAssumeCapacity(.{ .partially_bound_bit = desc.partially_bound });
    }

    // Translate the push constant ranges
    var pc_ranges: std.BoundedArray(vk.PushConstantRange, global_options.combined_pipeline_layout_create_buf_len) = .{};
    var pc_offset: u32 = 0;
    for (options.push_constant_ranges) |range| {
        // Add the range
        pc_ranges.append(.{
            .stage_flags = .{
                .vertex_bit = range.stages.vertex,
                .fragment_bit = range.stages.fragment,
                .compute_bit = range.stages.compute,
            },
            .offset = pc_offset,
            .size = range.size,
        }) catch @panic("OOB");
        pc_offset += range.size;
    }

    var binding_flags: vk.DescriptorSetLayoutBindingFlagsCreateInfo = .{
        .binding_count = @intCast(flags.len),
        .p_binding_flags = flags.constSlice().ptr,
    };

    const descriptor_set_layout = self.backend.device.createDescriptorSetLayout(&.{
        .binding_count = @intCast(descs.len),
        .p_bindings = &descs.buffer,
        .p_next = &binding_flags,
    }, null) catch @panic("OOM");
    setName(self.backend.debug_messenger, self.backend.device, descriptor_set_layout, options.name);

    // Create the pipeline layout
    const pipeline_layout = self.backend.device.createPipelineLayout(&.{
        .set_layout_count = 1,
        .p_set_layouts = &.{descriptor_set_layout},
        .push_constant_range_count = @intCast(pc_ranges.len),
        .p_push_constant_ranges = pc_ranges.constSlice().ptr,
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.debug_messenger, self.backend.device, pipeline_layout, options.name);

    return .{
        .desc_set = .fromBackendType(descriptor_set_layout),
        .handle = .fromBackendType(pipeline_layout),
    };
}

pub fn pipelineLayoutDestroy(
    self: *Gx,
    layout: gpu.Pipeline.Layout,
) void {
    self.backend.device.destroyPipelineLayout(layout.handle.asBackendType(), null);
    self.backend.device.destroyDescriptorSetLayout(layout.desc_set.asBackendType(), null);
}

pub fn cmdBufBeginZone(self: *Gx, cb: gpu.CmdBuf, loc: *const tracy.SourceLocation) void {
    if (self.backend.debug_messenger != .null_handle) {
        self.backend.device.cmdBeginDebugUtilsLabelEXT(cb.asBackendType(), &.{
            .p_label_name = loc.name orelse loc.function,
            .color = .{
                @as(f32, @floatFromInt(loc.color.r)) / 255.0,
                @as(f32, @floatFromInt(loc.color.g)) / 255.0,
                @as(f32, @floatFromInt(loc.color.b)) / 255.0,
                @as(f32, @floatFromInt(loc.color.a)) / 255.0,
            },
        });
    }
    if (tracy.enabled and self.timestamp_queries) {
        const query_id: gpu.CmdBuf.TracyQueryId = .next(self);
        self.device.tracy_queue.beginZone(.{
            .query_id = @bitCast(query_id),
            .loc = loc,
        });
        self.backend.device.cmdWriteTimestamp(
            cb.asBackendType(),
            .{ .top_of_pipe_bit = true },
            self.backend.tracy_query_pools[self.frame],
            query_id.index,
        );
    }
}

pub fn cmdBufEndZone(self: *Gx, cb: gpu.CmdBuf) void {
    if (self.backend.debug_messenger != .null_handle) {
        self.backend.device.cmdEndDebugUtilsLabelEXT(cb.asBackendType());
    }

    if (tracy.enabled and self.timestamp_queries) {
        const query_id: gpu.CmdBuf.TracyQueryId = .next(self);
        self.backend.device.cmdWriteTimestamp(
            cb.asBackendType(),
            .{ .bottom_of_pipe_bit = true },
            self.backend.tracy_query_pools[self.frame],
            query_id.index,
        );
        self.device.tracy_queue.endZone(@bitCast(query_id));
    }
}

pub fn cmdBufCreate(
    self: *Gx,
    loc: *const tracy.SourceLocation,
) gpu.CmdBuf {
    var cbs = [_]vk.CommandBuffer{.null_handle};
    self.backend.device.allocateCommandBuffers(&.{
        .command_pool = self.backend.cmd_pools[self.frame],
        .level = .primary,
        .command_buffer_count = cbs.len,
    }, &cbs) catch |err| @panic(@errorName(err));
    const cb = cbs[0];
    setName(self.backend.debug_messenger, self.backend.device, cb, .{
        .str = loc.name orelse loc.function,
    });

    self.backend.device.beginCommandBuffer(cb, &.{
        .flags = .{ .one_time_submit_bit = true },
        .p_inheritance_info = null,
    }) catch |err| @panic(@errorName(err));

    return .fromBackendType(cb);
}

pub fn cmdBufBeginRendering(
    self: *Gx,
    cb: gpu.CmdBuf,
    options: gpu.CmdBuf.BeginRenderingOptions,
) void {
    const color_attachments = gpu.Attachment.asBackendSlice(options.color_attachments);
    self.backend.device.cmdBeginRendering(cb.asBackendType(), &.{
        .flags = .{},
        .render_area = .{
            .offset = .{ .x = options.area.offset.x, .y = options.area.offset.y },
            .extent = .{
                .width = options.area.extent.width,
                .height = options.area.extent.height,
            },
        },
        .layer_count = 1,
        .view_mask = 0,
        .color_attachment_count = @intCast(color_attachments.len),
        .p_color_attachments = color_attachments.ptr,
        .p_depth_attachment = @ptrCast(options.depth_attachment),
        .p_stencil_attachment = @ptrCast(options.stencil_attachment),
    });
}

pub fn cmdBufEndRendering(self: *Gx, cb: gpu.CmdBuf) void {
    self.backend.device.cmdEndRendering(cb.asBackendType());
}

pub fn cmdBufPushConstants(
    self: *Gx,
    cb: gpu.CmdBuf,
    options: gpu.CmdBuf.PushConstantSliceOptions,
) void {
    self.backend.device.cmdPushConstants(
        cb.asBackendType(),
        options.pipeline_layout.asBackendType(),
        .{
            .vertex_bit = options.stages.vertex,
            .fragment_bit = options.stages.fragment,
            .compute_bit = options.stages.compute,
        },
        options.offset,
        @intCast(options.data.len * @sizeOf(u32)),
        options.data.ptr,
    );
}

pub fn cmdBufDraw(
    self: *Gx,
    cb: gpu.CmdBuf,
    options: gpu.CmdBuf.DrawOptions,
) void {
    self.backend.device.cmdDraw(
        cb.asBackendType(),
        options.vertex_count,
        options.instance_count,
        options.first_vertex,
        options.first_instance,
    );
}

pub fn cmdBufDispatch(
    self: *Gx,
    cb: gpu.CmdBuf,
    options: gpu.CmdBuf.DispatchOptions,
) void {
    self.backend.device.cmdDispatch(cb.asBackendType(), options.x, options.y, options.z);
}

pub fn cmdBufSetViewport(
    self: *Gx,
    cb: gpu.CmdBuf,
    viewport: gpu.Viewport,
) void {
    self.backend.device.cmdSetViewport(cb.asBackendType(), 0, 1, &.{.{
        .x = viewport.x,
        .y = viewport.y,
        .width = viewport.width,
        .height = viewport.height,
        .min_depth = viewport.min_depth,
        .max_depth = viewport.max_depth,
    }});
}

pub fn cmdBufSetScissor(
    self: *Gx,
    cb: gpu.CmdBuf,
    scissor: gpu.Rect2D,
) void {
    self.backend.device.cmdSetScissor(cb.asBackendType(), 0, 1, &.{.{
        .offset = .{
            .x = scissor.offset.x,
            .y = scissor.offset.y,
        },
        .extent = .{
            .width = scissor.extent.width,
            .height = scissor.extent.height,
        },
    }});
}

fn pipelineKindToVk(self: gpu.Pipeline.Kind) vk.PipelineBindPoint {
    return switch (self) {
        .graphics => .graphics,
        .compute => .compute,
    };
}

pub fn cmdBufBindPipeline(
    self: *Gx,
    cb: gpu.CmdBuf,
    pipeline: gpu.Pipeline,
) void {
    self.backend.device.cmdBindPipeline(
        cb.asBackendType(),
        pipelineKindToVk(pipeline.kind),
        pipeline.handle.asBackendType(),
    );
}

pub fn cmdBufBindDescSet(
    self: *Gx,
    cb: gpu.CmdBuf,
    pipeline: gpu.Pipeline,
    set: gpu.DescSet,
) void {
    self.backend.device.cmdBindDescriptorSets(
        cb.asBackendType(),
        pipelineKindToVk(pipeline.kind),
        pipeline.layout.asBackendType(),
        0,
        1,
        &.{set.asBackendType()},
        0,
        &[0]u32{},
    );
}

pub fn cmdBufPrepareSubmit(
    self: *Gx,
    cb: gpu.CmdBuf,
) void {
    cb.endZone(self);
    self.backend.device.endCommandBuffer(cb.asBackendType()) catch |err| @panic(@errorName(err));
}

pub fn cmdBufSubmit(
    self: *Gx,
    cb: gpu.CmdBuf,
) void {
    cmdBufPrepareSubmit(self, cb);
    const queue_submit_zone = Zone.begin(.{ .name = "queue submit", .src = @src() });
    defer queue_submit_zone.end();
    const cbs = [_]vk.CommandBuffer{cb.asBackendType()};
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

pub fn descPoolDestroy(self: *Gx, pool: gpu.DescPool) void {
    self.backend.device.destroyDescriptorPool(pool.asBackendType(), null);
}

pub fn descPoolCreate(self: *Gx, options: gpu.DescPool.Options) gpu.DescPool {
    // Create the descriptor pool
    const desc_pool = b: {
        // Calculate the size of the pool
        var samplers: u32 = 0;
        var combined_image_samplers: u32 = 0;
        var sampled_images: u32 = 0;
        var storage_images: u32 = 0;
        var uniform_buffers: u32 = 0;
        var storage_buffers: u32 = 0;
        var descriptors: u32 = 0;

        for (options.cmds) |cmd| {
            for (cmd.layout_options.descs) |desc| {
                switch (desc.kind) {
                    .sampler => samplers += desc.count,
                    .combined_image_sampler => combined_image_samplers += desc.count,
                    .sampled_image => sampled_images += desc.count,
                    .storage_image => storage_images += desc.count,
                    .uniform_buffer => uniform_buffers += desc.count,
                    .storage_buffer => storage_buffers += desc.count,
                }
            }
            descriptors += @intCast(cmd.layout_options.descs.len);
        }

        // Descriptor count must be greater than zero, so skip any that are zero
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorPoolSize.html
        var sizes: std.BoundedArray(vk.DescriptorPoolSize, 4) = .{};
        if (samplers > 0) sizes.appendAssumeCapacity(.{
            .type = .sampler,
            .descriptor_count = samplers,
        });
        if (combined_image_samplers > 0) sizes.appendAssumeCapacity(.{
            .type = .combined_image_sampler,
            .descriptor_count = combined_image_samplers,
        });
        if (sampled_images > 0) sizes.appendAssumeCapacity(.{
            .type = .sampled_image,
            .descriptor_count = sampled_images,
        });
        if (storage_images > 0) sizes.appendAssumeCapacity(.{
            .type = .storage_image,
            .descriptor_count = storage_images,
        });
        if (uniform_buffers > 0) sizes.appendAssumeCapacity(.{
            .type = .uniform_buffer,
            .descriptor_count = uniform_buffers,
        });
        if (storage_buffers > 0) sizes.appendAssumeCapacity(.{
            .type = .storage_buffer,
            .descriptor_count = storage_buffers,
        });

        // Create the descriptor pool
        const desc_pool = self.backend.device.createDescriptorPool(&.{
            .pool_size_count = @intCast(sizes.len),
            .p_pool_sizes = &sizes.buffer,
            .flags = .{},
            .max_sets = @intCast(options.cmds.len),
        }, null) catch |err| @panic(@errorName(err));
        setName(self.backend.debug_messenger, self.backend.device, desc_pool, options.name);

        break :b desc_pool;
    };

    // Create the descriptor sets
    {
        // Collect the arguments for descriptor set creation
        var layout_buf: std.BoundedArray(vk.DescriptorSetLayout, global_options.init_desc_pool_buf_len) = .{};
        var results: [global_options.init_desc_pool_buf_len]vk.DescriptorSet = undefined;
        for (options.cmds) |cmd| {
            layout_buf.appendAssumeCapacity(cmd.layout.asBackendType());
        }

        // Allocate the descriptor sets
        self.backend.device.allocateDescriptorSets(&.{
            .descriptor_pool = desc_pool,
            .descriptor_set_count = @intCast(layout_buf.len),
            .p_set_layouts = &layout_buf.buffer,
        }, &results) catch |err| @panic(@errorName(err));

        // Write the results
        for (options.cmds, results[0..options.cmds.len]) |cmd, result| {
            cmd.result.* = .fromBackendType(result);
            setName(self.backend.debug_messenger, self.backend.device, result, cmd.name);
        }
    }

    // Return the descriptor pool
    return .fromBackendType(desc_pool);
}

pub fn descSetsUpdate(self: *Gx, updates: []const gpu.DescSet.Update) void {
    const buf_len = global_options.update_desc_sets_buf_len;

    var buffer_infos: std.BoundedArray(vk.DescriptorBufferInfo, buf_len) = .{};
    var image_infos: std.BoundedArray(vk.DescriptorImageInfo, buf_len) = .{};
    var write_sets: std.BoundedArray(vk.WriteDescriptorSet, buf_len) = .{};

    // Iterate over the updates
    var i: u32 = 0;
    while (i < updates.len) {
        // Find all subsequent updates on the same set binding and type
        const batch_first_update = updates[i];
        const batch_set = batch_first_update.set;
        const batch_binding = batch_first_update.binding;
        const batch_kind: gpu.DescSet.Update.Value.Tag = batch_first_update.value;
        const batch_index_start: u32 = batch_first_update.index;
        var batch_size: u32 = 0;
        while (true) {
            const update_curr = updates[i + batch_size];

            switch (update_curr.value) {
                .sampler => |sampler| {
                    image_infos.appendAssumeCapacity(.{
                        .sampler = sampler.asBackendType(),
                        .image_view = .null_handle,
                        .image_layout = .undefined,
                    });
                },
                .combined_image_sampler => |combined| {
                    image_infos.appendAssumeCapacity(.{
                        .sampler = combined.sampler.asBackendType(),
                        .image_view = combined.view.asBackendType(),
                        .image_layout = .read_only_optimal,
                    });
                },
                .sampled_image => |view| {
                    image_infos.appendAssumeCapacity(.{
                        .sampler = .null_handle,
                        .image_view = view.asBackendType(),
                        .image_layout = .read_only_optimal,
                    });
                },
                .storage_image => |view| {
                    image_infos.appendAssumeCapacity(.{
                        .sampler = .null_handle,
                        .image_view = view.asBackendType(),
                        .image_layout = .general,
                    });
                },
                .uniform_buf => |view| buffer_infos.appendAssumeCapacity(.{
                    .buffer = view.handle.asBackendType(),
                    .offset = view.offset,
                    .range = view.len,
                }),
                .storage_buf => |view| buffer_infos.appendAssumeCapacity(.{
                    .buffer = view.handle.asBackendType(),
                    .offset = view.offset,
                    .range = view.len,
                }),
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
            .sampler => {
                const batch_combined_image_samplers = image_infos.constSlice()[image_infos.len - batch_size ..];
                write_sets.appendAssumeCapacity(.{
                    .dst_set = batch_set.asBackendType(),
                    .dst_binding = batch_binding,
                    .dst_array_element = batch_index_start,
                    .descriptor_type = .sampler,
                    .descriptor_count = batch_size,
                    .p_buffer_info = &[0]vk.DescriptorBufferInfo{},
                    .p_image_info = batch_combined_image_samplers.ptr,
                    .p_texel_buffer_view = &[0]vk.BufferView{},
                });
            },
            .combined_image_sampler => {
                const batch_combined_image_samplers = image_infos.constSlice()[image_infos.len - batch_size ..];
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
            .sampled_image => {
                const batch_sampled_images = image_infos.constSlice()[image_infos.len - batch_size ..];
                write_sets.appendAssumeCapacity(.{
                    .dst_set = batch_set.asBackendType(),
                    .dst_binding = batch_binding,
                    .dst_array_element = batch_index_start,
                    .descriptor_type = .sampled_image,
                    .descriptor_count = batch_size,
                    .p_buffer_info = &[0]vk.DescriptorBufferInfo{},
                    .p_image_info = batch_sampled_images.ptr,
                    .p_texel_buffer_view = &[0]vk.BufferView{},
                });
            },
            .storage_image => {
                const batch_sampled_images = image_infos.constSlice()[image_infos.len - batch_size ..];
                write_sets.appendAssumeCapacity(.{
                    .dst_set = batch_set.asBackendType(),
                    .dst_binding = batch_binding,
                    .dst_array_element = batch_index_start,
                    .descriptor_type = .storage_image,
                    .descriptor_count = batch_size,
                    .p_buffer_info = &[0]vk.DescriptorBufferInfo{},
                    .p_image_info = batch_sampled_images.ptr,
                    .p_texel_buffer_view = &[0]vk.BufferView{},
                });
            },
            .uniform_buf => {
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
            .storage_buf => {
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
        }

        i += batch_size;
    }

    self.backend.device.updateDescriptorSets(@intCast(write_sets.len), &write_sets.buffer, 0, null);
}

pub fn acquireNextImage(self: *Gx, framebuf_extent: gpu.Extent2D) gpu.ImageView.Sized2D {
    // Acquire the image
    const acquire_result = b: {
        const acquire_zone = Zone.begin(.{
            .src = @src(),
            .name = "acquire next image",
        });
        defer acquire_zone.end();
        if (self.backend.swapchain.out_of_date) {
            self.backend.swapchain.recreate(self, framebuf_extent);
        }
        while (true) {
            break :b self.backend.device.acquireNextImageKHR(
                self.backend.swapchain.swapchain,
                std.math.maxInt(u64),
                self.backend.image_availables[self.frame],
                .null_handle,
            ) catch |err| switch (err) {
                error.OutOfDateKHR, error.FullScreenExclusiveModeLostEXT => {
                    self.backend.swapchain.recreate(self, framebuf_extent);
                    continue;
                },
                error.OutOfHostMemory,
                error.OutOfDeviceMemory,
                error.Unknown,
                error.SurfaceLostKHR,
                error.DeviceLost,
                => @panic(@errorName(err)),
            };
        }
    };
    assert(self.backend.image_index == null);
    self.backend.image_index = acquire_result.image_index;

    // Transition it to the right format
    {
        const transition_zone = Zone.begin(.{ .name = "prepare swapchain image", .src = @src() });
        defer transition_zone.end();

        const cb: gpu.CmdBuf = .init(self, .{
            .name = "Prepare Swapchain Image",
            .src = @src(),
        });

        transitionImageToColorAttachmentOptimal(
            self,
            cb.asBackendType(),
            self.backend.swapchain.images.get(acquire_result.image_index),
        );

        cmdBufPrepareSubmit(self, cb);
        self.backend.device.queueSubmit(
            self.backend.queue,
            1,
            &.{.{
                .wait_semaphore_count = 1,
                .p_wait_semaphores = &.{self.backend.image_availables[self.frame]},
                .p_wait_dst_stage_mask = &.{.{ .top_of_pipe_bit = true }},
                .command_buffer_count = 1,
                .p_command_buffers = &.{cb.asBackendType()},
                .signal_semaphore_count = 0,
                .p_signal_semaphores = &.{},
                .p_next = null,
            }},
            .null_handle,
        ) catch |err| @panic(@errorName(err));
    }

    return .{
        .view = .fromBackendType(self.backend.swapchain.views.get(acquire_result.image_index)),
        .extent = .{
            .width = self.backend.swapchain.swap_extent.width,
            .height = self.backend.swapchain.swap_extent.height,
        },
    };
}

pub fn beginFrame(self: *Gx) void {
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
    const cmd_pool = &self.backend.cmd_pools[self.frame];
    self.backend.device.resetCommandPool(cmd_pool.*, .{}) catch |err| @panic(@errorName(err));
    if (self.validate) {
        // https://github.com/Games-by-Mason/gpu/issues/3
        self.backend.device.destroyCommandPool(cmd_pool.*, null);
        cmd_pool.* = self.backend.device.createCommandPool(&.{
            .flags = .{ .transient_bit = true },
            .queue_family_index = self.backend.queue_family_index,
        }, null) catch |err| @panic(@errorName(err));
    }
    reset_cmd_pool_zone.end();

    if (tracy.enabled and self.timestamp_queries) {
        const tracy_query_pool_zone = Zone.begin(.{
            .src = @src(),
            .name = "tracy query pool",
        });
        defer tracy_query_pool_zone.end();

        const queries = self.tracy_queries[self.frame];
        if (queries > 0) {
            var results: [gpu.CmdBuf.TracyQueryId.cap * 2]u64 = undefined;
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
                        .query_id = @bitCast(gpu.CmdBuf.TracyQueryId{
                            .index = @intCast(i),
                            .frame = self.frame,
                        }),
                        .gpu_time = time,
                    });
                }
            }

            // We can't use our command buffer abstraction here because it issues queries
            var cbs = [_]vk.CommandBuffer{.null_handle};
            self.backend.device.allocateCommandBuffers(&.{
                .command_pool = self.backend.cmd_pools[self.frame],
                .level = .primary,
                .command_buffer_count = cbs.len,
            }, &cbs) catch |err| @panic(@errorName(err));
            const cb = cbs[0];
            setName(self.backend.debug_messenger, self.backend.device, cb, .{
                .str = "Clear Tracy Query Pool",
            });
            self.backend.device.beginCommandBuffer(cb, &.{
                .flags = .{ .one_time_submit_bit = true },
                .p_inheritance_info = null,
            }) catch |err| @panic(@errorName(err));
            self.backend.device.cmdResetQueryPool(
                cb,
                self.backend.tracy_query_pools[self.frame],
                0,
                gpu.CmdBuf.TracyQueryId.cap,
            );
            self.backend.device.endCommandBuffer(cb) catch |err| @panic(@errorName(err));
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

fn imageOptionsToVk(options: btypes.ImageOptions) vk.ImageCreateInfo {
    return .{
        .flags = .{
            .cube_compatible_bit = options.flags.cube_compatible,
            .@"2d_array_compatible_bit" = options.flags.@"2d_array_compatible",
        },
        .image_type = switch (options.dimensions) {
            .@"1d", .@"1d_array" => .@"1d",
            .@"2d", .@"2d_array", .cube, .cube_array => .@"2d",
            .@"3d" => .@"3d",
        },
        .format = options.format.asBackendType(),
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
        .tiling = .optimal,
        .usage = .{
            .transfer_src_bit = options.usage.transfer_src,
            .transfer_dst_bit = options.usage.transfer_dst,
            .sampled_bit = options.usage.sampled,
            .storage_bit = options.usage.storage,
            .color_attachment_bit = options.usage.color_attachment,
            .depth_stencil_attachment_bit = options.usage.depth_stencil_attachment,
            .input_attachment_bit = options.usage.input_attachment,
        },
        .sharing_mode = .exclusive,
        .queue_family_index_count = 0,
        .p_queue_family_indices = null,
        .initial_layout = .undefined,
    };
}

fn createImageView(
    self: *Gx,
    name: gpu.DebugName,
    image: vk.Image,
    options: btypes.ImageOptions,
) gpu.ImageView {
    const view = self.backend.device.createImageView(&.{
        .image = image,
        .view_type = switch (options.dimensions) {
            .@"1d" => .@"1d",
            .@"2d" => .@"2d",
            .@"3d" => .@"3d",
            .cube => .cube,
            .@"1d_array" => .@"1d_array",
            .@"2d_array" => .@"2d_array",
            .cube_array => .cube_array,
        },
        .format = options.format.asBackendType(),
        .components = .{
            .r = .identity,
            .g = .identity,
            .b = .identity,
            .a = .identity,
        },
        .subresource_range = .{
            .aspect_mask = aspectToVk(options.aspect),
            .base_mip_level = 0,
            .level_count = options.mip_levels,
            .base_array_layer = 0,
            .layer_count = options.array_layers,
        },
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.debug_messenger, self.backend.device, view, name);
    return .fromBackendType(view);
}

pub fn imageCreateDedicated(
    self: *Gx,
    name: gpu.DebugName,
    options: btypes.ImageOptions,
) gpu.Image(.any).InitDedicatedResult {
    // Create the image
    const image = self.backend.device.createImage(&imageOptionsToVk(options), null) catch |err| @panic(@errorName(err));
    setName(self.backend.debug_messenger, self.backend.device, image, name);
    var reqs2: vk.MemoryRequirements2 = .{ .memory_requirements = undefined };

    // Get the memory requirements
    self.backend.device.getImageMemoryRequirements2(&.{ .image = image }, &reqs2);
    const reqs = reqs2.memory_requirements;
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
    setName(self.backend.debug_messenger, self.backend.device, memory, name);

    // Bind the image to the memory
    self.backend.device.bindImageMemory(
        image,
        memory,
        0,
    ) catch |err| @panic(@errorName(err));

    // Return the image and dedicated memory
    return .{
        .image = .{
            .handle = .fromBackendType(image),
            .view = createImageView(self, name, image, options),
        },
        .memory = .{
            .handle = .fromBackendType(memory),
            .size = reqs.size,
        },
    };
}

pub fn imageCreatePlaced(
    self: *Gx,
    name: gpu.DebugName,
    memory: gpu.MemoryHandle,
    offset: u64,
    options: btypes.ImageOptions,
) gpu.Image(.any) {
    // Create the image
    const image = self.backend.device.createImage(&imageOptionsToVk(options), null) catch |err| @panic(@errorName(err));
    setName(self.backend.debug_messenger, self.backend.device, image, name);

    // Place the image
    self.backend.device.bindImageMemory(
        image,
        memory.asBackendType(),
        offset,
    ) catch |err| @panic(@errorName(err));

    // Return the image
    return .{
        .handle = .fromBackendType(image),
        .view = createImageView(self, name, image, options),
    };
}

pub fn imageViewDestroy(self: *Gx, view: gpu.ImageView) void {
    self.backend.device.destroyImageView(view.asBackendType(), null);
}

pub fn imageDestroy(self: *Gx, image: gpu.ImageHandle) void {
    self.backend.device.destroyImage(image.asBackendType(), null);
}

pub fn imageMemoryRequirements(
    self: *Gx,
    options: btypes.ImageOptions,
) gpu.MemoryRequirements {
    // Get the image options
    const options_vk = imageOptionsToVk(options);

    // Panic if this image format isn't supported, `vkGetDeviceImageMemoryRequirements` won't check
    // this for us
    var props: vk.ImageFormatProperties2 = .{
        .image_format_properties = undefined,
    };
    self.backend.instance.getPhysicalDeviceImageFormatProperties2(
        self.backend.physical_device.device,
        &.{
            .format = options_vk.format,
            .type = options_vk.image_type,
            .tiling = options_vk.tiling,
            .usage = options_vk.usage,
            .flags = options_vk.flags,
        },
        &props,
    ) catch |err| @panic(@errorName(err));

    // Get the memory requirements
    var dedicated_reqs: vk.MemoryDedicatedRequirements = .{
        .prefers_dedicated_allocation = vk.FALSE,
        .requires_dedicated_allocation = vk.FALSE,
    };
    var reqs2: vk.MemoryRequirements2 = .{
        .memory_requirements = undefined,
        .p_next = &dedicated_reqs,
    };
    self.backend.device.getDeviceImageMemoryRequirements(&.{
        .p_create_info = &options_vk,
        .plane_aspect = .{},
    }, &reqs2);
    const reqs = reqs2.memory_requirements;
    return .{
        .size = reqs.size,
        .alignment = reqs.alignment,
        .dedicated = if (dedicated_reqs.requires_dedicated_allocation == vk.TRUE)
            .required
        else if (dedicated_reqs.prefers_dedicated_allocation == vk.TRUE)
            .preferred
        else
            .discouraged,
    };
}

pub fn memoryCreate(self: *Gx, options: btypes.MemoryCreateOptions) gpu.MemoryHandle {
    const memory_type_bits = switch (options.usage) {
        .color_image => b: {
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
                        .format = .r8g8b8a8_srgb, // Supported by all DX12 hardware
                        .extent = .{
                            .width = 16,
                            .height = 16,
                            .depth = 1,
                        },
                        .mip_levels = 1,
                        .array_layers = 1,
                        .samples = .{ .@"1_bit" = true },
                        .tiling = .optimal,
                        .usage = .{ .sampled_bit = true },
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
        .depth_stencil_image => |format| b: {
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
                        .format = format.asBackendType(),
                        .extent = .{
                            .width = 16,
                            .height = 16,
                            .depth = 1,
                        },
                        .mip_levels = 1,
                        .array_layers = 1,
                        .samples = .{ .@"1_bit" = true },
                        .tiling = .optimal,
                        .usage = .{ .sampled_bit = true },
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
    setName(self.backend.debug_messenger, self.backend.device, memory, options.name);
    return .fromBackendType(memory);
}

pub fn memoryDestroy(self: *Gx, memory: gpu.MemoryHandle) void {
    self.backend.device.freeMemory(memory.asBackendType(), null);
}

pub fn pipelineDestroy(self: *Gx, pipeline: gpu.Pipeline) void {
    self.backend.device.destroyPipeline(pipeline.handle.asBackendType(), null);
}

pub fn shaderModuleCreate(self: *Gx, options: gpu.ShaderModule.Options) gpu.ShaderModule {
    const module = self.backend.device.createShaderModule(&.{
        .code_size = options.ir.len * @sizeOf(u32),
        .p_code = options.ir.ptr,
    }, null) catch |err| @panic(@errorName(err));
    setName(
        self.backend.debug_messenger,
        self.backend.device,
        module,
        options.name,
    );
    return .fromBackendType(module);
}

pub fn shaderModuleDestroy(self: *Gx, module: gpu.ShaderModule) void {
    self.backend.device.destroyShaderModule(module.asBackendType(), null);
}

pub fn pipelinesCreateGraphics(self: *Gx, cmds: []const gpu.Pipeline.InitGraphicsCmd) void {
    // Settings that are constant across all our pipelines
    const dynamic_states = [_]vk.DynamicState{
        .viewport,
        .scissor,
    };
    const dynamic_state: vk.PipelineDynamicStateCreateInfo = .{
        .dynamic_state_count = dynamic_states.len,
        .p_dynamic_states = &dynamic_states,
    };
    const vertex_input: vk.PipelineVertexInputStateCreateInfo = .{
        .vertex_binding_description_count = 0,
        .p_vertex_binding_descriptions = &.{},
        .vertex_attribute_description_count = 0,
        .p_vertex_attribute_descriptions = &.{},
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
    const max_shader_stages = gpu.Pipeline.InitGraphicsCmd.Stages.max_stages;
    var shader_stages: std.BoundedArray(vk.PipelineShaderStageCreateInfo, global_options.init_pipelines_buf_len * max_shader_stages) = .{};
    var pipeline_infos: std.BoundedArray(vk.GraphicsPipelineCreateInfo, global_options.init_pipelines_buf_len) = .{};
    var input_assemblys: std.BoundedArray(vk.PipelineInputAssemblyStateCreateInfo, global_options.init_pipelines_buf_len) = .{};
    var rendering_infos: std.BoundedArray(vk.PipelineRenderingCreateInfo, global_options.init_desc_pool_buf_len) = .{};
    for (cmds) |cmd| {
        const input_assembly = input_assemblys.addOneAssumeCapacity();
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

        shader_stages.appendAssumeCapacity(.{
            .stage = .{ .vertex_bit = true },
            .module = cmd.stages.vertex.asBackendType(),
            .p_name = "main",
        });
        shader_stages.appendAssumeCapacity(.{
            .stage = .{ .fragment_bit = true },
            .module = cmd.stages.fragment.asBackendType(),
            .p_name = "main",
        });
        const shader_stages_slice = shader_stages.constSlice()[shader_stages.len - 2 ..];

        const rendering_info = rendering_infos.addOneAssumeCapacity();
        const color_attachment_formats = gpu.ImageFormat.asBackendSlice(cmd.color_attachment_formats);
        rendering_info.* = .{
            .view_mask = 0,
            .color_attachment_count = @intCast(color_attachment_formats.len),
            .p_color_attachment_formats = color_attachment_formats.ptr,
            .depth_attachment_format = cmd.depth_attachment_format.asBackendType(),
            .stencil_attachment_format = cmd.stencil_attachment_format.asBackendType(),
        };

        pipeline_infos.appendAssumeCapacity(.{
            .stage_count = @intCast(shader_stages_slice.len),
            .p_stages = shader_stages_slice.ptr,
            .p_vertex_input_state = &vertex_input,
            .p_input_assembly_state = input_assembly,
            .p_viewport_state = &viewport_state,
            .p_rasterization_state = &rasterizer,
            .p_multisample_state = &multisampling,
            .p_depth_stencil_state = null,
            .p_color_blend_state = &color_blending,
            .p_dynamic_state = &dynamic_state,
            .layout = cmd.layout.handle.asBackendType(),
            .render_pass = .null_handle,
            .subpass = 0,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = -1,
            .p_next = rendering_info,
        });
    }

    // Create the pipelines
    var pipelines: [global_options.init_pipelines_buf_len]vk.Pipeline = undefined;
    const create_result = self.backend.device.createGraphicsPipelines(
        self.backend.pipeline_cache,
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
        setName(self.backend.debug_messenger, self.backend.device, pipeline, cmd.name);
        cmd.result.* = .{
            .layout = cmd.layout.handle,
            .handle = .fromBackendType(pipeline),
            .kind = .graphics,
        };
    }
}

pub fn pipelinesCreateCompute(self: *Gx, cmds: []const gpu.Pipeline.InitComputeCmd) void {
    var pipeline_infos: std.BoundedArray(vk.ComputePipelineCreateInfo, global_options.init_pipelines_buf_len) = .{};
    for (cmds) |cmd| {
        pipeline_infos.appendAssumeCapacity(.{
            .flags = .{},
            .stage = .{
                .stage = .{ .compute_bit = true },
                .module = cmd.shader_module.asBackendType(),
                .p_name = "main",
            },
            .layout = cmd.layout.handle.asBackendType(),
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = -1,
        });
    }

    // Create the pipelines
    var pipelines: [global_options.init_pipelines_buf_len]vk.Pipeline = undefined;
    const create_result = self.backend.device.createComputePipelines(
        self.backend.pipeline_cache,
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
        setName(self.backend.debug_messenger, self.backend.device, pipeline, cmd.name);
        cmd.result.* = .{
            .layout = cmd.layout.handle,
            .handle = .fromBackendType(pipeline),
            .kind = .compute,
        };
    }
}

fn transitionImageColorAttachmentToPresent(
    self: *Gx,
    cb: vk.CommandBuffer,
    image: vk.Image,
) void {
    self.backend.device.cmdPipelineBarrier2(cb, &.{
        .dependency_flags = .{},
        .memory_barrier_count = 0,
        .p_memory_barriers = &.{},
        .buffer_memory_barrier_count = 0,
        .p_buffer_memory_barriers = &.{},
        .image_memory_barrier_count = 1,
        .p_image_memory_barriers = &.{.{
            .src_stage_mask = .{ .color_attachment_output_bit = true },
            .src_access_mask = .{ .color_attachment_write_bit = true },
            .dst_stage_mask = .{ .bottom_of_pipe_bit = true },
            .dst_access_mask = .{},
            .old_layout = .attachment_optimal,
            .new_layout = .present_src_khr,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }},
    });
}

fn transitionImageToColorAttachmentOptimal(
    self: *Gx,
    cb: vk.CommandBuffer,
    image: vk.Image,
) void {
    self.backend.device.cmdPipelineBarrier2(cb, &.{
        .dependency_flags = .{},
        .memory_barrier_count = 0,
        .p_memory_barriers = &.{},
        .buffer_memory_barrier_count = 0,
        .p_buffer_memory_barriers = &.{},
        .image_memory_barrier_count = 1,
        .p_image_memory_barriers = &.{.{
            .src_stage_mask = .{ .top_of_pipe_bit = true },
            .src_access_mask = .{},
            .dst_stage_mask = .{ .color_attachment_output_bit = true },
            .dst_access_mask = .{ .color_attachment_write_bit = true },
            .old_layout = .undefined,
            .new_layout = .attachment_optimal,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }},
    });
}

pub fn endFrame(self: *Gx, options: Gx.EndFrameOptions) void {
    if (!options.present) {
        // We aren't presenting, just wrap up this command pool submission by signaling the fence
        // for this frame and then early out.
        self.backend.device.queueSubmit(
            self.backend.queue,
            1,
            &.{.{
                .wait_semaphore_count = 0,
                .p_wait_semaphores = &.{},
                .p_wait_dst_stage_mask = &.{},
                .command_buffer_count = 0,
                .p_command_buffers = &.{},
                .signal_semaphore_count = 0,
                .p_signal_semaphores = &.{},
                .p_next = null,
            }},
            self.backend.cmd_pool_ready[self.frame],
        ) catch |err| @panic(@errorName(err));
        return;
    }

    {
        const transition_zone = Zone.begin(.{ .name = "finalize swapchain image", .src = @src() });
        defer transition_zone.end();

        const cb: gpu.CmdBuf = .init(self, .{
            .name = "Finalize Swapchain Image",
            .src = @src(),
        });

        transitionImageColorAttachmentToPresent(
            self,
            cb.asBackendType(),
            self.backend.swapchain.images.get(self.backend.image_index.?),
        );

        cmdBufPrepareSubmit(self, cb);
        self.backend.device.queueSubmit(
            self.backend.queue,
            1,
            &.{.{
                .wait_semaphore_count = 0,
                .p_wait_semaphores = &.{},
                .p_wait_dst_stage_mask = &.{},
                .command_buffer_count = 1,
                .p_command_buffers = &.{cb.asBackendType()},
                .signal_semaphore_count = 1,
                .p_signal_semaphores = &.{
                    self.backend.ready_for_present[self.backend.image_index.?],
                },
                .p_next = null,
            }},
            self.backend.cmd_pool_ready[self.frame],
        ) catch |err| @panic(@errorName(err));
    }

    {
        const queue_present_zone = Zone.begin(.{ .name = "queue present", .src = @src() });
        defer queue_present_zone.end();
        const swapchain = [_]vk.SwapchainKHR{self.backend.swapchain.swapchain};
        const image_index = [_]u32{self.backend.image_index.?};

        const result = self.backend.device.queuePresentKHR(
            self.backend.queue,
            &.{
                .wait_semaphore_count = 1,
                .p_wait_semaphores = &.{self.backend.ready_for_present[self.backend.image_index.?]},
                .swapchain_count = swapchain.len,
                .p_swapchains = &swapchain,
                .p_image_indices = &image_index,
                .p_results = null,
            },
        ) catch |err| b: switch (err) {
            error.OutOfDateKHR, error.FullScreenExclusiveModeLostEXT => {
                self.backend.swapchain.out_of_date = true;
                break :b .success;
            },
            error.OutOfHostMemory,
            error.OutOfDeviceMemory,
            error.Unknown,
            error.SurfaceLostKHR,
            error.DeviceLost,
            => @panic(@errorName(err)),
        };
        if (result == .suboptimal_khr) {
            self.backend.swapchain.out_of_date = true;
        }
    }
}

pub fn samplerCreate(
    self: *Gx,
    name: gpu.DebugName,
    options: gpu.Sampler.Options,
) gpu.Sampler {
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
        .anisotropy_enable = @intFromBool(options.max_anisotropy != .none and self.backend.physical_device.sampler_anisotropy),
        .max_anisotropy = @min(@as(f32, @floatFromInt(@as(u8, @intFromEnum(options.max_anisotropy)))), self.backend.physical_device.max_sampler_anisotropy),
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
        // Can be useful, but I believe not supported as a sampler option by DX12. Use `texelFetch`
        // in the shader instead.
        .unnormalized_coordinates = vk.FALSE,
    }, null) catch |err| @panic(@errorName(err));
    setName(self.backend.debug_messenger, self.backend.device, sampler, name);
    return .fromBackendType(sampler);
}

pub fn samplerDestroy(self: *Gx, sampler: gpu.Sampler) void {
    self.backend.device.destroySampler(sampler.asBackendType(), null);
}

pub const TimestampCalibration = struct {
    cpu: u64,
    gpu: u64,
    max_deviation: u64,

    fn init(device: vk.DeviceProxy, timestamp_queries: bool) TimestampCalibration {
        if (!timestamp_queries) return .{
            .cpu = 0,
            .gpu = 0,
            .max_deviation = 0,
        };
        var calibration_results: [2]u64 = undefined;
        const max_deviation = device.getCalibratedTimestampsKHR(
            2,
            &.{
                .{ .time_domain = switch (builtin.os.tag) {
                    .windows => .query_performance_counter_ext,
                    else => .clock_monotonic_raw_khr,
                } },
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
};

fn rangeToVk(range: gpu.ImageBarrier.Range, aspect: gpu.ImageAspect) vk.ImageSubresourceRange {
    return .{
        .aspect_mask = aspectToVk(aspect),
        .base_mip_level = range.base_mip_level,
        .level_count = range.mip_levels,
        .base_array_layer = range.base_array_layer,
        .layer_count = range.array_layers,
    };
}

pub fn imageBarrierUndefinedToTransferDst(
    options: gpu.ImageBarrier.UndefinedToTransferDstOptions,
) gpu.ImageBarrier {
    return .{ .backend = .{
        .src_stage_mask = .{ .top_of_pipe_bit = true },
        .src_access_mask = .{},
        .dst_stage_mask = .{ .all_transfer_bit = true },
        .dst_access_mask = .{ .transfer_write_bit = true },
        .old_layout = .undefined,
        .new_layout = .transfer_dst_optimal,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = options.handle.asBackendType(),
        .subresource_range = rangeToVk(options.range, options.aspect),
    } };
}

pub fn imageBarrierUndefinedToColorAttachment(
    options: gpu.ImageBarrier.UndefinedToColorAttachmentOptions,
) gpu.ImageBarrier {
    return .{ .backend = .{
        .src_stage_mask = .{ .top_of_pipe_bit = true },
        .src_access_mask = .{},
        .dst_stage_mask = .{ .color_attachment_output_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true },
        .old_layout = .undefined,
        .new_layout = .attachment_optimal,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = options.handle.asBackendType(),
        .subresource_range = rangeToVk(options.range, .{ .color = true }),
    } };
}

pub fn imageBarrierUndefinedToColorAttachmentAfterRead(
    options: gpu.ImageBarrier.UndefinedToColorAttachmentOptionsAfterRead,
) gpu.ImageBarrier {
    return .{ .backend = .{
        .src_stage_mask = .{
            .vertex_shader_bit = options.src_stage.vertex_shader,
            .fragment_shader_bit = options.src_stage.fragment_shader,
        },
        .src_access_mask = .{ .shader_read_bit = true },
        .dst_stage_mask = .{ .color_attachment_output_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true },
        .old_layout = .undefined,
        .new_layout = .attachment_optimal,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = options.handle.asBackendType(),
        .subresource_range = rangeToVk(options.range, .{ .color = true }),
    } };
}

pub fn imageBarrierTransferDstToReadOnly(
    options: gpu.ImageBarrier.TransferDstToReadOnlyOptions,
) gpu.ImageBarrier {
    return .{ .backend = .{
        .src_stage_mask = .{ .copy_bit = true },
        .src_access_mask = .{ .transfer_write_bit = true },
        .dst_stage_mask = .{
            .vertex_shader_bit = options.dst_stage.vertex_shader,
            .fragment_shader_bit = options.dst_stage.fragment_shader,
        },
        .dst_access_mask = .{ .shader_read_bit = true },
        .old_layout = .transfer_dst_optimal,
        .new_layout = .read_only_optimal,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = options.handle.asBackendType(),
        .subresource_range = rangeToVk(options.range, options.aspect),
    } };
}

pub fn imageBarrierTransferDstToColorAttachment(
    options: gpu.ImageBarrier.TransferDstToColorAttachmentOptions,
) gpu.ImageBarrier {
    return .{ .backend = .{
        .src_stage_mask = .{ .copy_bit = true },
        .src_access_mask = .{ .transfer_write_bit = true },
        .dst_stage_mask = .{ .color_attachment_output_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true },
        .old_layout = .transfer_dst_optimal,
        .new_layout = .attachment_optimal,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = options.handle.asBackendType(),
        .subresource_range = rangeToVk(options.range, .{ .color = true }),
    } };
}

pub fn imageBarrierReadOnlyToColorAttachment(
    options: gpu.ImageBarrier.ReadOnlyToColorAttachmentOptions,
) gpu.ImageBarrier {
    return .{ .backend = .{
        .src_stage_mask = .{
            .vertex_shader_bit = options.src_stage.vertex_shader,
            .fragment_shader_bit = options.src_stage.fragment_shader,
        },
        .src_access_mask = .{ .shader_read_bit = true },
        .dst_stage_mask = .{ .color_attachment_output_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true },
        .old_layout = .read_only_optimal,
        .new_layout = .attachment_optimal,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = options.handle.asBackendType(),
        .subresource_range = rangeToVk(options.range, .{ .color = true }),
    } };
}

pub fn imageBarrierColorAttachmentToReadOnly(
    options: gpu.ImageBarrier.ColorAttachmentToReadOnlyOptions,
) gpu.ImageBarrier {
    return .{ .backend = .{
        .src_stage_mask = .{ .color_attachment_output_bit = true },
        .src_access_mask = .{ .color_attachment_write_bit = true },
        .dst_stage_mask = .{
            .vertex_shader_bit = options.dst_stage.vertex_shader,
            .fragment_shader_bit = options.dst_stage.fragment_shader,
        },
        .dst_access_mask = .{ .shader_read_bit = true },
        .old_layout = .attachment_optimal,
        .new_layout = .read_only_optimal,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = options.handle.asBackendType(),
        .subresource_range = rangeToVk(options.range, options.aspect),
    } };
}

pub fn imageBarrierColorAttachmentToCompute(
    options: gpu.ImageBarrier.ColorAttachmentToComputeOptions,
) gpu.ImageBarrier {
    return .{ .backend = .{
        .src_stage_mask = .{ .color_attachment_output_bit = true },
        .src_access_mask = .{ .color_attachment_write_bit = true },
        .dst_stage_mask = .{ .compute_shader_bit = true },
        .dst_access_mask = .{
            .shader_read_bit = options.dst_access.read,
            .shader_write_bit = options.dst_access.write,
        },
        .old_layout = .attachment_optimal,
        .new_layout = .general,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = options.handle.asBackendType(),
        .subresource_range = rangeToVk(options.range, .{ .color = true }),
    } };
}

pub fn imageBarrierComputeToColorAttachment(
    options: gpu.ImageBarrier.ComputeToColorAttachmentOptions,
) gpu.ImageBarrier {
    return .{ .backend = .{
        .src_stage_mask = .{ .compute_shader_bit = true },
        .src_access_mask = .{
            .shader_read_bit = options.src_access.read,
            .shader_write_bit = options.src_access.write,
        },
        .dst_stage_mask = .{ .color_attachment_output_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true },
        .old_layout = .general,
        .new_layout = .attachment_optimal,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = options.handle.asBackendType(),
        .subresource_range = rangeToVk(options.range, .{ .color = true }),
    } };
}

pub fn imageBarrierComputeToReadOnly(
    options: gpu.ImageBarrier.ComputeToReadOnlyOptions,
) gpu.ImageBarrier {
    return .{ .backend = .{
        .src_stage_mask = .{ .compute_shader_bit = true },
        .src_access_mask = .{
            .shader_read_bit = options.src_access.read,
            .shader_write_bit = options.src_access.write,
        },
        .dst_stage_mask = .{
            .vertex_shader_bit = options.dst_stage.vertex_shader,
            .fragment_shader_bit = options.dst_stage.fragment_shader,
        },
        .dst_access_mask = .{ .shader_read_bit = true },
        .old_layout = .general,
        .new_layout = .read_only_optimal,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = options.handle.asBackendType(),
        .subresource_range = rangeToVk(options.range, options.aspect),
    } };
}

pub fn bufBarrierComputeWriteToGraphicsRead(
    options: gpu.BufBarrier.ComputeWriteToGraphicsReadOptions,
) gpu.BufBarrier {
    return .{
        .backend = .{
            .src_stage_mask = .{ .compute_shader_bit = true },
            .src_access_mask = .{ .shader_write_bit = true },
            .dst_stage_mask = .{
                .vertex_shader_bit = options.dst_stage.vertex_shader,
                .fragment_shader_bit = options.dst_stage.fragment_shader,
            },
            .dst_access_mask = .{ .shader_read_bit = true },
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .buffer = options.handle.asBackendType(),
            .offset = 0,
            // I'm under the impression that in practice GPUs aren't using this value, so we're not
            // going to complicate the API by asking for it. If this turns out to be incorrect we can
            // always add it later, but this is further implied by the fact that AFAICT DX12 doesn't
            // have a way to pass this value in.
            .size = vk.WHOLE_SIZE,
        },
    };
}

pub fn bufBarrierComputeReadToGraphicsWrite(
    options: gpu.BufBarrier.ComputeReadToGraphicsWriteOptions,
) gpu.BufBarrier {
    return .{
        .backend = .{
            .src_stage_mask = .{ .compute_shader_bit = true },
            .src_access_mask = .{ .shader_read_bit = true },
            .dst_stage_mask = .{
                .vertex_shader_bit = options.dst_stage.vertex_shader,
                .fragment_shader_bit = options.dst_stage.fragment_shader,
            },
            .dst_access_mask = .{ .shader_write_bit = true },
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .buffer = options.handle.asBackendType(),
            .offset = 0,
            // See `bufBarrierComputeWriteToGraphicsRead`.
            .size = vk.WHOLE_SIZE,
        },
    };
}

pub fn bufBarrierGraphicsReadToComputeWrite(
    options: gpu.BufBarrier.GraphicsReadToComputeWriteOptions,
) gpu.BufBarrier {
    return .{
        .backend = .{
            .src_stage_mask = .{
                .vertex_shader_bit = options.src_stage.vertex_shader,
                .fragment_shader_bit = options.src_stage.fragment_shader,
            },
            .src_access_mask = .{ .shader_read_bit = true },
            .dst_stage_mask = .{ .compute_shader_bit = true },
            .dst_access_mask = .{ .shader_write_bit = true },
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .buffer = options.handle.asBackendType(),
            .offset = 0,
            // See `bufBarrierComputeWriteToGraphicsRead`.
            .size = vk.WHOLE_SIZE,
        },
    };
}

pub fn bufBarrierGraphicsWriteToComputeRead(
    options: gpu.BufBarrier.GraphicsWriteToComputeReadOptions,
) gpu.BufBarrier {
    return .{
        .backend = .{
            .src_stage_mask = .{
                .vertex_shader_bit = options.src_stage.vertex_shader,
                .fragment_shader_bit = options.src_stage.fragment_shader,
            },
            .src_access_mask = .{ .shader_write_bit = true },
            .dst_stage_mask = .{ .compute_shader_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .buffer = options.handle.asBackendType(),
            .offset = 0,
            // See `bufBarrierComputeWriteToGraphicsRead`.
            .size = vk.WHOLE_SIZE,
        },
    };
}

pub fn cmdBufBarriers(
    self: *Gx,
    cb: gpu.CmdBuf,
    options: gpu.CmdBuf.BarriersOptions,
) void {
    const image_barriers = gpu.ImageBarrier.asBackendSlice(options.image);
    const buffer_barriers = gpu.BufBarrier.asBackendSlice(options.buffer);
    self.backend.device.cmdPipelineBarrier2(cb.asBackendType(), &.{
        .dependency_flags = .{},
        .memory_barrier_count = 0,
        .p_memory_barriers = &.{},
        .buffer_memory_barrier_count = @intCast(buffer_barriers.len),
        .p_buffer_memory_barriers = buffer_barriers.ptr,
        .image_memory_barrier_count = @intCast(image_barriers.len),
        .p_image_memory_barriers = image_barriers.ptr,
    });
}

pub fn imageUploadRegionInit(options: gpu.ImageUpload.Region.Options) gpu.ImageUpload.Region {
    return .{ .backend = .{
        .buffer_offset = options.buffer_offset,
        .buffer_row_length = options.buffer_row_length orelse 0,
        .buffer_image_height = options.buffer_image_height orelse 0,
        .image_subresource = .{
            .aspect_mask = aspectToVk(options.aspect),
            .mip_level = options.mip_level,
            .base_array_layer = options.base_array_layer,
            .layer_count = options.array_layers,
        },
        .image_offset = .{
            .x = options.image_offset.x,
            .y = options.image_offset.y,
            .z = options.image_offset.z,
        },
        .image_extent = .{
            .width = options.image_extent.width,
            .height = options.image_extent.height,
            .depth = options.image_extent.depth,
        },
    } };
}

pub fn bufferUploadRegionInit(options: Gx.BufferUpload.Region.Options) Gx.BufferUpload.Region {
    return .{ .backend = .{
        .src_offset = options.src_offset,
        .dst_offset = options.dst_offset,
        .size = options.size,
    } };
}

pub fn attachmentInit(options: gpu.Attachment.Options) gpu.Attachment {
    return .{ .backend = .{
        .image_view = options.view.asBackendType(),
        .image_layout = .attachment_optimal,
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
    } };
}

pub fn cmdBufUploadImage(
    self: *Gx,
    cb: gpu.CmdBuf,
    dst: gpu.ImageHandle,
    src: gpu.BufHandle(.{}),
    regions: []const gpu.ImageUpload.Region,
) void {
    // `cmdCopyImage` has been superseded by `cmdCopyImage2`, however there's no benefit to the
    // new API unless you need a `pNext` chain, and as such we're opting to just save the extra
    // bytes.
    const vk_regions = gpu.ImageUpload.Region.asBackendSlice(regions);
    self.backend.device.cmdCopyBufferToImage(
        cb.asBackendType(),
        src.asBackendType(),
        dst.asBackendType(),
        .transfer_dst_optimal,
        @intCast(vk_regions.len),
        vk_regions.ptr,
    );
}

pub fn cmdBufUploadBuffer(
    self: *Gx,
    cb: gpu.CmdBuf,
    dst: Gx.Buf(.{}),
    src: Gx.Buf(.{}),
    regions: []const Gx.BufferUpload.Region,
) void {
    // `cmdCopyBuffer` has been superseded by `cmdCopyBuffer2`, however there's no benefit to the
    // new API unless you need a `pNext` chain, and as such we're opting to just save the extra
    // bytes.
    const vk_regions = Gx.BufferUpload.Region.asBackendSlice(regions);
    self.backend.device.cmdCopyBuffer(
        cb.asBackendType(),
        src.asBackendType(),
        dst.asBackendType(),
        @intCast(vk_regions.len),
        vk_regions.ptr,
    );
}

pub fn waitIdle(self: *const Gx) void {
    self.backend.device.deviceWaitIdle() catch |err| @panic(@errorName(err));
}

fn aspectToVk(self: gpu.ImageAspect) vk.ImageAspectFlags {
    return .{
        .color_bit = self.color,
        .depth_bit = self.depth,
        .stencil_bit = self.stencil,
    };
}

fn filterToVk(filter: gpu.Sampler.Options.Filter) vk.Filter {
    return switch (filter) {
        .nearest => .nearest,
        .linear => .linear,
    };
}

fn addressModeToVk(mode: gpu.Sampler.Options.AddressMode) vk.SamplerAddressMode {
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
    access: btypes.MemoryCreateOptions.Access,
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
    pub const empty: @This() = .{
        .swapchain = .null_handle,
        .images = .{},
        .views = .{},
        .swap_extent = .{
            .width = 0,
            .height = 0,
        },
        .external_framebuf_size = .{
            .width = 0,
            .height = 0,
        },
        .out_of_date = true,
    };

    swapchain: vk.SwapchainKHR,
    images: std.BoundedArray(vk.Image, max_swapchain_depth),
    views: std.BoundedArray(vk.ImageView, max_swapchain_depth),
    swap_extent: vk.Extent2D,
    external_framebuf_size: gpu.Extent2D,
    out_of_date: bool = false,

    fn init(
        instance: vk.InstanceProxy,
        framebuf_extent: gpu.Extent2D,
        device: vk.DeviceProxy,
        physical_device: PhysicalDevice,
        surface: vk.SurfaceKHR,
        old_swapchain: vk.SwapchainKHR,
        debug_messenger: vk.DebugUtilsMessengerEXT,
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
            break :e vk.Extent2D{
                .width = std.math.clamp(
                    framebuf_extent.width,
                    surface_capabilities.min_image_extent.width,
                    surface_capabilities.max_image_extent.width,
                ),
                .height = std.math.clamp(
                    framebuf_extent.height,
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
        setName(debug_messenger, device, swapchain, .{ .str = "Main" });
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
            setName(debug_messenger, device, image, .{ .str = "Swapchain", .index = i });
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
            setName(debug_messenger, device, view, .{ .str = "Swapchain", .index = i });
            views.appendAssumeCapacity(view);
        }

        return .{
            .swapchain = swapchain,
            .images = images,
            .views = views,
            .swap_extent = swap_extent,
            .external_framebuf_size = framebuf_extent,
        };
    }

    fn destroyEverythingExceptSwapchain(self: *@This(), device: vk.DeviceProxy) void {
        for (self.views.constSlice()) |v| device.destroyImageView(v, null);
    }

    fn deinit(self: *@This(), device: vk.DeviceProxy) void {
        self.destroyEverythingExceptSwapchain(device);
        device.destroySwapchainKHR(self.swapchain, null);
        self.* = undefined;
    }

    fn recreate(self: *@This(), gx: *Gx, framebuf_extent: gpu.Extent2D) void {
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
        if (self.swapchain != .null_handle) {
            gx.backend.device.deviceWaitIdle() catch |err| std.debug.panic("vkDeviceWaitIdle failed: {}", .{err});
        }
        const retired = self.swapchain;
        self.destroyEverythingExceptSwapchain(gx.backend.device);
        self.* = .init(
            gx.backend.instance,
            framebuf_extent,
            gx.backend.device,
            gx.backend.physical_device,
            gx.backend.surface,
            retired,
            gx.backend.debug_messenger,
        );
        if (self.swapchain != .null_handle) {
            gx.backend.device.destroySwapchainKHR(retired, null);
        }
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
    ty: gpu.Device.Kind = undefined,
    min_uniform_buffer_offset_alignment: u16 = undefined,
    min_storage_buffer_offset_alignment: u16 = undefined,
    min_texel_buffer_offset_alignment: u16 = undefined,
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
    try writer.print("\t* type: {}\n", .{data.message_type});

    if (data.data) |d| {
        try writer.writeAll("\t* id: ");
        if (d.*.p_message_id_name) |name| {
            try writer.print("{s}", .{name});
        } else {
            try writer.writeAll("null");
        }
        try writer.print(" ({})\n", .{d.*.message_id_number});

        if (d.*.p_message) |message| {
            try writer.print("\t* message: {s}", .{message});
        }

        if (d.*.queue_label_count > 0) {
            try writer.writeByte('\n');
            if (d.*.p_queue_labels) |queue_labels| {
                for (queue_labels[0..d.*.queue_label_count]) |label| {
                    try writer.print("\t* queue: {s}\n", .{label.p_label_name});
                }
            }
        }

        if (d.*.cmd_buf_label_count > 0) {
            try writer.writeByte('\n');
            if (d.*.p_cmd_buf_labels) |cmd_buf_labels| {
                for (cmd_buf_labels[0..d.*.cmd_buf_label_count]) |label| {
                    try writer.print("\t* command buffer: {s}\n", .{label.p_label_name});
                }
            }
        }

        if (d.*.object_count > 0) {
            try writer.writeByte('\n');
            if (d.*.p_objects) |objects| {
                for (objects[0..d.*.object_count], 0..) |object, object_i| {
                    try writer.print("\t* object {}:\n", .{object_i});

                    try writer.writeAll("\t\t* name: ");
                    if (object.p_object_name) |name| {
                        try writer.print("{s}", .{name});
                    } else {
                        try writer.writeAll("null");
                    }
                    try writer.writeByte('\n');

                    try writer.print("\t\t* type: {}\n", .{object.object_type});
                    try writer.print("\t\t* handle: 0x{x}", .{object.object_handle});
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
) callconv(vk.vulkan_call_conv) vk.Bool32 {
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

    // Ignore or reduce the severity of some warnings
    if (level == .warn) {
        if (data) |d| switch (d.*.message_id_number) {
            // Ignore `BestPractices-vkCreateDevice-physical-device-features-not-retrieved`, this is
            // a false positive--we're using `vkGetPhysicalDeviceFeatures2`.
            584333584 => return vk.FALSE,
            // Ignore `BestPractices-vkBindBufferMemory-small-dedicated-allocation` and
            // `BestPractices-vkAllocateMemory-small-allocation`, our whole rendering strategy is
            // designed around this but we often have so little total memory that we trip it anyway!
            280337739, -40745094 => return vk.FALSE,
            // Don't warn us that validation is on every time validation is on, but do log it as
            // debug
            615892639, 2132353751, 1734198062 => level = .debug,
            // Don't warn us that the swapchain is out of date, we handle this it's not an
            // exceptional situation!
            1762589289 => level = .debug,
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
            @panic("validation error");
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
    debug_messenger: vk.DebugUtilsMessengerEXT,
    device: vk.DeviceProxy,
    object: anytype,
    debug_name: gpu.DebugName,
) void {
    if (debug_messenger == .null_handle) return;

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

fn appendNext(head: *?*vk.BaseInStructure, new: *vk.BaseInStructure) void {
    assert(new.p_next == null);
    new.p_next = head.*;
    head.* = new;
}

pub const OsStr = struct {
    pub const Ptr = switch (builtin.os.tag) {
        .windows => std.os.windows.LPCWSTR,
        else => [*:0]const u8,
    };

    pub const Optional = struct {
        ptr: ?Ptr,

        pub const none: @This() = .{ .ptr = null };

        pub fn fromLit(comptime opt: ?[:0]const u8) @This() {
            const str = opt orelse return .none;
            return OsStr.fromLit(str).optional();
        }

        pub fn unwrap(self: @This()) ?OsStr {
            const ptr = self.ptr orelse return null;
            return .{ .ptr = ptr };
        }
    };

    ptr: Ptr,

    pub fn fromLit(comptime str: [:0]const u8) @This() {
        return .{ .ptr = switch (builtin.os.tag) {
            .windows => std.unicode.utf8ToUtf16LeStringLiteral(str),
            else => str,
        } };
    }

    pub fn optional(self: @This()) Optional {
        return .{ .ptr = self.ptr };
    }
};

fn setenv(name: OsStr, value: OsStr) void {
    if (builtin.os.tag == .windows) {
        if (std.os.windows.kernel32.SetEnvironmentVariableW(name.ptr, value.ptr) == 0) {
            @panic("SetEnvironmentVariableW failed");
        }
    } else {
        const posix = struct {
            extern "c" fn setenv(
                name: [*:0]const u8,
                value: [*:0]const u8,
                overwrite: c_int,
            ) callconv(.c) c_int;
        };
        if (posix.setenv(name.ptr, value.ptr, 1) != 0) {
            @panic("setenv failed");
        }
    }
}

pub const Buf = vk.Buffer;
pub const CmdBuf = vk.CommandBuffer;
pub const DescPool = vk.DescriptorPool;
pub const DescSet = vk.DescriptorSet;
pub const DescSetLayout = vk.DescriptorSetLayout;
pub const Memory = vk.DeviceMemory;
pub const Image = vk.Image;
pub const ImageView = vk.ImageView;
pub const ShaderModule = vk.ShaderModule;
pub const Pipeline = vk.Pipeline;
pub const PipelineLayout = vk.PipelineLayout;
pub const Sampler = vk.Sampler;
pub const ImageBarrier = vk.ImageMemoryBarrier2;
pub const BufBarrier = vk.BufferMemoryBarrier2;
pub const ImageUploadRegion = vk.BufferImageCopy;
pub const BufferUploadRegion = vk.BufferCopy;
pub const Attachment = vk.RenderingAttachmentInfo;
pub const ImageFormat = vk.Format;

pub const named_image_formats: btypes.NamedImageFormats = .{
    .undefined = @intFromEnum(vk.Format.undefined),

    .r8_unorm = @intFromEnum(vk.Format.r8_unorm),
    .r8_snorm = @intFromEnum(vk.Format.r8_snorm),
    .r8_uint = @intFromEnum(vk.Format.r8_uint),
    .r8_sint = @intFromEnum(vk.Format.r8_sint),

    .r8g8b8a8_unorm = @intFromEnum(vk.Format.r8g8b8a8_unorm),
    .r8g8b8a8_snorm = @intFromEnum(vk.Format.r8g8b8a8_snorm),
    .r8g8b8a8_uint = @intFromEnum(vk.Format.r8g8b8a8_uint),
    .r8g8b8a8_sint = @intFromEnum(vk.Format.r8g8b8a8_sint),
    .r8g8b8a8_srgb = @intFromEnum(vk.Format.r8g8b8a8_srgb),

    .b8g8r8a8_unorm = @intFromEnum(vk.Format.b8g8r8a8_unorm),
    .b8g8r8a8_srgb = @intFromEnum(vk.Format.b8g8r8a8_srgb),

    .d24_unorm_s8_uint = @intFromEnum(vk.Format.d24_unorm_s8_uint),
    .d32_sfloat = @intFromEnum(vk.Format.d32_sfloat),
};
