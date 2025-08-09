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

// Context
surface: vk.SurfaceKHR,
base_wrapper: vk.BaseWrapper,
device: vk.DeviceProxy,
instance: vk.InstanceProxy,
physical_device: PhysicalDevice,
debug_messenger: vk.DebugUtilsMessengerEXT,
pipeline_cache: vk.PipelineCache,
surface_context: ?*anyopaque,

// Swapchain state
recreate_swapchain: bool,
swapchain: vk.SwapchainKHR,
swapchain_extent: gpu.Extent2D,
swapchain_images: std.ArrayListUnmanaged(vk.Image),
swapchain_views: std.ArrayListUnmanaged(vk.ImageView),
ready_for_present: std.ArrayListUnmanaged(vk.Semaphore),

// Queues & commands
timestamp_period: f32,
queue: vk.Queue,
queue_family_index: u32,
cmd_pools: [global_options.max_frames_in_flight]vk.CommandPool,

// Frame synchronization
image_availables: [global_options.max_frames_in_flight]vk.Semaphore,
cmd_pool_ready: [global_options.max_frames_in_flight]vk.Fence,

// Tracy info
tracy_query_pools: [global_options.max_frames_in_flight]vk.QueryPool,

pub const Options = struct {
    pub const CreateSurfaceError = error{
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
        surface_context: ?*anyopaque,
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

        // >99% of Windows and Linux devices in `vulkan.gpuinfo.org` support these features at the
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

/// We'd rather just have an array list of required extensions. Unfortunately, the calibrated
/// timestamps extension has two different names, and in the wild relevant devices sometimes support
/// only one or the other, so we need more logic here. If enough devices move to the KHR name
/// eventually then we can replace this whole thing with a simpl array list.
const DeviceExts = struct {
    khr_swapchain: bool = false,
    ext_hdr_metadata: bool = false,
    khr_calibrated_timestamps: bool = false,
    ext_calibrated_timestamps: bool = false,

    fn add(self: *@This(), ext: *const vk.ExtensionProperties) void {
        const name: []const u8 = std.mem.span(@as([*:0]const u8, @ptrCast(ext.extension_name[0..].ptr)));
        inline for (@typeInfo(@This()).@"struct".fields) |field| {
            if (std.mem.eql(u8, name, @field(vk.extensions, field.name).name)) {
                log.debug("\t* {s} v{} supported", .{ name, ext.spec_version });
                @field(self, field.name) = true;
            }
        }
    }

    fn alloc(
        self: @This(),
        gpa: Allocator,
        timestamp_queries: bool,
    ) ?[]const [*:0]const u8 {
        var result: std.ArrayListUnmanaged([*:0]const u8) = .{};

        // Required extensions
        if (self.khr_swapchain) {
            result.append(gpa, vk.extensions.khr_swapchain.name) catch @panic("OOM");
        } else {
            return null;
        }

        // Optional extensions
        if (self.ext_hdr_metadata) {
            result.append(gpa, vk.extensions.ext_hdr_metadata.name) catch @panic("OOM");
        }

        // Required if timestamp queries are enabled
        if (timestamp_queries) {
            if (self.khr_calibrated_timestamps) {
                result.append(gpa, vk.extensions.khr_calibrated_timestamps.name) catch @panic("OOM");
            } else if (self.ext_calibrated_timestamps) {
                result.append(gpa, vk.extensions.ext_calibrated_timestamps.name) catch @panic("OOM");
            } else {
                return null;
            }
        }

        return result.toOwnedSlice(gpa) catch @panic("OOM");
    }

    const GetCalibratedTimestampsFn = fn (
        self: vk.DeviceProxy,
        timestamp_count: u32,
        p_timestamp_infos: [*]const vk.CalibratedTimestampInfoKHR,
        p_timestamps: [*]u64,
    ) vk.DeviceProxy.GetCalibratedTimestampsKHRError!u64;

    fn getGetCalibratedTimestampsFn(self: @This()) ?*const GetCalibratedTimestampsFn {
        if (self.khr_calibrated_timestamps) {
            return &vk.DeviceProxy.getCalibratedTimestampsKHR;
        } else if (self.ext_calibrated_timestamps) {
            return &vk.DeviceProxy.getCalibratedTimestampsEXT;
        } else {
            return null;
        }
    }
};

const InstanceExts = struct {
    ext_debug_utils: bool = false,
    ext_swapchain_colorspace: bool = false,

    fn add(self: *@This(), ext: *const vk.ExtensionProperties) void {
        const name: []const u8 = std.mem.span(@as([*:0]const u8, @ptrCast(ext.extension_name[0..].ptr)));
        inline for (@typeInfo(@This()).@"struct".fields) |field| {
            if (std.mem.eql(u8, name, @field(vk.extensions, field.name).name)) {
                log.debug("{s} v{} supported", .{ name, ext.spec_version });
                @field(self, field.name) = true;
            }
        }
    }
};

pub fn init(
    gpa: Allocator,
    options: Gx.Options,
) btypes.BackendInitResult {
    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    // We don't use the scoped arena here because the amount of allocation we do is entirely up to
    // the user's computer. e.g. it could report more or less extensions, and we can't predict how
    // much memory is needed.
    var arena_allocator: std.heap.ArenaAllocator = .init(gpa);
    defer arena_allocator.deinit();
    const arena = arena_allocator.allocator();

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

    var required_instance_exts: std.ArrayListUnmanaged([*:0]const u8) = .{};
    required_instance_exts.appendSlice(arena, options.backend.instance_extensions) catch @panic("OOM");

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

    const enabled_validation_features = if (options.validation == .all)
        &enabled_validation_features_all
    else
        &enabled_validation_features_fast;

    var instance_validation_features: vk.ValidationFeaturesEXT = .{
        .enabled_validation_feature_count = @intCast(enabled_validation_features.len),
        .p_enabled_validation_features = enabled_validation_features.ptr,
    };

    var create_instance_chain: ?*vk.BaseInStructure = null;

    // Set requested layers, and log all in case any are implicit and end up causing problems
    {
        const val_layer_name = "VK_LAYER_KHRONOS_validation";
        const supported_layers = base_wrapper.enumerateInstanceLayerPropertiesAlloc(arena) catch |err| @panic(@errorName(err));
        var validation_layer_missing = options.validation.gte(.fast);

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

            if (options.validation.gte(.fast) and std.mem.eql(u8, curr_name, val_layer_name)) {
                appendNext(&create_instance_chain, @ptrCast(&instance_validation_features));
                layers.append(arena, val_layer_name) catch @panic("OOM");
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
    var instance_exts: InstanceExts = .{};
    {
        const supported_instance_exts = base_wrapper.enumerateInstanceExtensionPropertiesAlloc(
            null,
            arena,
        ) catch |err| @panic(@errorName(err));
        for (supported_instance_exts) |ext| instance_exts.add(&ext);
    }

    const debug = b: {
        if (!options.validation.gte(.minimal)) break :b false;
        if (instance_exts.ext_debug_utils) break :b true;
        log.warn("{s}: requested but not found", .{vk.extensions.ext_debug_utils.name});
        break :b false;
    };
    if (debug) {
        required_instance_exts.append(
            arena,
            vk.extensions.ext_debug_utils.name,
        ) catch @panic("OOM");
        appendNext(&create_instance_chain, @ptrCast(&instance_dbg_messenger_info));
    }
    if (instance_exts.ext_swapchain_colorspace) {
        required_instance_exts.append(
            arena,
            vk.extensions.ext_swapchain_colorspace.name,
        ) catch @panic("OOM");
    }

    log.debug("Required Instance Extensions:", .{});
    for (required_instance_exts.items) |name| {
        log.debug("* {s}", .{name});
    }
    log.debug("Required Layers:", .{});
    for (layers.items) |name| {
        log.debug("* {s}", .{name});
    }
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
        .enabled_extension_count = math.cast(u32, required_instance_exts.items.len) orelse @panic("overflow"),
        .pp_enabled_extension_names = required_instance_exts.items.ptr,
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
    const physical_devices = instance_proxy.enumeratePhysicalDevicesAlloc(arena) catch |err| @panic(@errorName(err));
    enumerate_devices_zone.end();

    var best_physical_device: PhysicalDevice = .{};

    log.info("All Devices:", .{});
    for (physical_devices, 0..) |device, i| {
        const properties = instance_proxy.getPhysicalDeviceProperties(device);
        log.info("  {}. {s}", .{ i, bufToStr(&properties.device_name) });
        log.debug("\t* device API version: {}", .{@as(vk.Version, @bitCast(properties.api_version))});
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

        const queue_family_properties = instance_proxy.getPhysicalDeviceQueueFamilyPropertiesAlloc(device, arena) catch |err| @panic(@errorName(err));
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

        var device_exts: DeviceExts = .{};
        const supported_device_extensions = instance_proxy.enumerateDeviceExtensionPropertiesAlloc(device, null, arena) catch |err| @panic(@errorName(err));
        for (supported_device_extensions) |extension_properties| {
            device_exts.add(&extension_properties);
        }

        const device_ext_list = device_exts.alloc(arena, options.timestamp_queries);

        const surface_capabilities, const surface_format, const present_mode = if (device_ext_list != null) b: {
            const surface_capabilities = instance_proxy.getPhysicalDeviceSurfaceCapabilitiesKHR(
                device,
                surface,
            ) catch |err| @panic(@errorName(err));

            const supported_surface_formats = instance_proxy.getPhysicalDeviceSurfaceFormatsAllocKHR(
                device,
                surface,
                arena,
            ) catch |err| @panic(@errorName(err));

            log.debug("    * supported surface formats:", .{});
            for (supported_surface_formats) |supported| {
                log.debug("        * {}, {}", .{ supported.color_space, supported.format });
            }
            const surface_format: ?gpu.SurfaceFormatQuery.Result = sf: {
                for (options.surface_format) |query| {
                    for (query.image_formats) |query_format| {
                        for (supported_surface_formats) |supported| {
                            // Only check for the alternate color spaces if they're actually supported
                            if (supported.color_space != .srgb_nonlinear_khr and !instance_exts.ext_swapchain_colorspace) {
                                log.warn(
                                    "{s} formats found but extension unsupported",
                                    .{vk.extensions.ext_swapchain_colorspace.name},
                                );
                                continue;
                            }

                            // Check that we're in the right color space
                            if (supported.color_space != query.color_space.asBackendType()) continue;

                            // Check that our usage flags are supported. According to vulkan.gpuinfo.org, 100%
                            // of GPUs surveyed support these usages, but we still want to check just in case--
                            // and since it's unclear whether that means every surface supports them or at least
                            // one surface format does.
                            if (!surface_capabilities.supported_usage_flags.color_attachment_bit) continue;
                            if (!surface_capabilities.supported_usage_flags.transfer_dst_bit) continue;

                            // Check that we're the right format
                            if (query_format.asBackendType() == supported.format) {
                                break :sf .{
                                    .color_space = query.color_space,
                                    .image_format = .fromBackendType(supported.format),
                                    .userdata = query.userdata,
                                };
                            }
                        }
                    }
                }
                break :sf null;
            };
            var best_present_mode: vk.PresentModeKHR = .fifo_khr;
            const present_modes = instance_proxy.getPhysicalDeviceSurfacePresentModesAllocKHR(
                device,
                surface,
                arena,
            ) catch |err| @panic(@errorName(err));
            for (present_modes) |present_mode| {
                switch (present_mode) {
                    .fifo_relaxed_khr => best_present_mode = present_mode,
                    else => {},
                }
            }
            break :b .{ surface_capabilities, surface_format, best_present_mode };
        } else .{ null, null, null };

        log.debug("\t* best surface format: {?}", .{surface_format});
        log.debug("\t* present mode: {?}", .{present_mode});
        log.debug("\t* device extensions: {}", .{device_exts});

        const composite_alpha: ?vk.CompositeAlphaFlagsKHR = b: {
            if (surface_capabilities) |sc| {
                const supported = sc.supported_composite_alpha;
                log.debug("\t* supported composite alpha: {any}", .{supported});
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
        const compatible = version_compatible and queue_family_index != null and device_ext_list != null and composite_alpha != null and supports_required_features;

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
                .device_exts = device_exts,
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
    const queue_family_properties = instance_proxy.getPhysicalDeviceQueueFamilyPropertiesAlloc(best_physical_device.device, arena) catch |err| @panic(@errorName(err));

    const queue_family_allocated: []u8 = arena.alloc(u8, queue_family_properties.len) catch @panic("OOM");
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
    const device_exts = best_physical_device.device_exts.alloc(arena, options.timestamp_queries).?;
    const device_create_info: vk.DeviceCreateInfo = .{
        .p_queue_create_infos = &queue_create_infos,
        .queue_create_info_count = @intCast(queue_create_infos.len),
        .p_enabled_features = null,
        .enabled_extension_count = @intCast(device_exts.len),
        .pp_enabled_extension_names = device_exts.ptr,
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

    var ready_for_present = std.ArrayListUnmanaged(vk.Semaphore)
        .initCapacity(gpa, options.max_swapchain_images) catch @panic("OOM");
    for (0..ready_for_present.capacity) |frame| {
        const semaphore = device.createSemaphore(
            &.{},
            null,
        ) catch |err| @panic(@errorName(err));
        setName(debug_messenger, device, semaphore, .{
            .str = "Ready For Present",
            .index = frame,
        });
        ready_for_present.appendAssumeCapacity(semaphore);
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

    const calibration: TimestampCalibration = .init(best_physical_device.device_exts, device, options.timestamp_queries);
    const tracy_queue = TracyQueue.init(.{
        .gpu_time = calibration.gpu,
        .period = timestamp_period,
        .context = 0,
        .flags = .{},
        .type = .vulkan,
        .name = graphics_queue_name,
    });

    // Create the backend result
    var result: btypes.BackendInitResult = .{
        .backend = .{
            .surface = surface,
            .base_wrapper = base_wrapper,
            .debug_messenger = debug_messenger,
            .pipeline_cache = pipeline_cache,
            .instance = instance_proxy,
            .device = device,
            .swapchain = .null_handle,
            .recreate_swapchain = false,
            .swapchain_images = std.ArrayListUnmanaged(vk.Image)
                .initCapacity(gpa, options.max_swapchain_images) catch @panic("OOM"),
            .swapchain_views = std.ArrayListUnmanaged(vk.ImageView)
                .initCapacity(gpa, options.max_swapchain_images) catch @panic("OOM"),
            .swapchain_extent = .{ .width = 0, .height = 0 },
            .cmd_pools = cmd_pools,
            .image_availables = image_availables,
            .ready_for_present = ready_for_present,
            .cmd_pool_ready = cmd_pool_ready,
            .physical_device = best_physical_device,
            .timestamp_period = timestamp_period,
            .queue = queue,
            .queue_family_index = best_physical_device.queue_family_index,
            .tracy_query_pools = tracy_query_pools,
            .surface_context = options.backend.surface_context,
        },
        .device = .{
            .kind = best_physical_device.ty,
            .uniform_buf_offset_alignment = best_physical_device.min_uniform_buffer_offset_alignment,
            .storage_buf_offset_alignment = best_physical_device.min_storage_buffer_offset_alignment,
            .texel_buffer_offset_alignment = best_physical_device.min_texel_buffer_offset_alignment,
            .timestamp_period = timestamp_period,
            .tracy_queue = tracy_queue,
            .surface_format = best_physical_device.surface_format,
        },
    };

    // Set up the swapchain
    result.backend.setSwapchainExtent(options.surface_extent, null);

    return result;
}

pub fn deinit(self: *Gx, gpa: Allocator) void {
    // Destroy the pipeline cache
    self.backend.device.destroyPipelineCache(self.backend.pipeline_cache, null);

    // Destroy the Tracy data
    for (self.backend.tracy_query_pools) |pool| {
        self.backend.device.destroyQueryPool(pool, null);
    }

    // Destroy internal sync state
    for (self.backend.ready_for_present.items) |semaphore| {
        self.backend.device.destroySemaphore(semaphore, null);
    }
    self.backend.ready_for_present.deinit(gpa);
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
    destroySwapchainViewsAndResetImages(&self.backend);
    self.backend.swapchain_views.deinit(gpa);
    self.backend.swapchain_images.deinit(gpa);
    self.backend.device.destroySwapchainKHR(self.backend.swapchain, null);

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
    options: gpu.Pipeline.Layout.InitOptions,
) gpu.Pipeline.Layout {
    const arena = self.arena.begin() catch @panic("OOM");
    defer self.arena.end();

    // Create the descriptor set layout
    const descs = arena.alloc(vk.DescriptorSetLayoutBinding, options.layout.descs.len) catch @panic("OOM");
    const flags = arena.alloc(vk.DescriptorBindingFlags, options.layout.descs.len) catch @panic("OOM");
    for (descs, flags, options.layout.descs, 0..) |*desc, *flag, input, i| {
        desc.* = .{
            .binding = @intCast(i),
            .descriptor_type = switch (input.kind) {
                .sampler => .sampler,
                .sampled_image => .sampled_image,
                .storage_image => .storage_image,
                .uniform_buffer => .uniform_buffer,
                .storage_buffer => .storage_buffer,
            },
            .descriptor_count = input.count,
            .stage_flags = shaderStagesToVk(input.stages),
            .p_immutable_samplers = b: {
                // For each sampler, check if there's an immutable sampler set for it. This sounds
                // bad from a complexity standpoint, but it would be difficult to actually bind
                // enough samplers for this to be an issue.
                if (input.kind == .sampler) {
                    for (options.immutable_samplers) |is| {
                        if (is.binding == i) {
                            break :b @ptrCast(is.samplers);
                        }
                    }
                }
                break :b null;
            },
        };
        flag.* = .{ .partially_bound_bit = input.partially_bound };
    }

    // Translate the push constant ranges
    const pc_ranges = arena.alloc(vk.PushConstantRange, options.layout.push_constant_ranges.len) catch @panic("OOM");
    var pc_offset: u32 = 0;
    for (pc_ranges, options.layout.push_constant_ranges) |*range, input| {
        // Add the range
        range.* = .{
            .stage_flags = shaderStagesToVk(input.stages),
            .offset = pc_offset,
            .size = input.size,
        };
        pc_offset += input.size;
    }

    var binding_flags: vk.DescriptorSetLayoutBindingFlagsCreateInfo = .{
        .binding_count = @intCast(flags.len),
        .p_binding_flags = flags.ptr,
    };

    const descriptor_set_layout = self.backend.device.createDescriptorSetLayout(&.{
        .binding_count = @intCast(descs.len),
        .p_bindings = descs.ptr,
        .p_next = &binding_flags,
    }, null) catch @panic("OOM");
    setName(
        self.backend.debug_messenger,
        self.backend.device,
        descriptor_set_layout,
        options.layout.name,
    );

    // Create the pipeline layout
    const pipeline_layout = self.backend.device.createPipelineLayout(&.{
        .set_layout_count = 1,
        .p_set_layouts = &.{descriptor_set_layout},
        .push_constant_range_count = @intCast(pc_ranges.len),
        .p_push_constant_ranges = pc_ranges.ptr,
    }, null) catch |err| @panic(@errorName(err));
    setName(
        self.backend.debug_messenger,
        self.backend.device,
        pipeline_layout,
        options.layout.name,
    );

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
            .{ .bottom_of_pipe_bit = true },
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

fn attachmentToVk(options: gpu.CmdBuf.BeginRenderingOptions.Attachment) vk.RenderingAttachmentInfo {
    return .{
        .image_view = options.view.asBackendType(),
        .image_layout = .attachment_optimal,
        .resolve_mode = switch (options.resolve_mode) {
            .none => .{},
            .sample_zero => .{ .sample_zero_bit = true },
            .average => .{ .average_bit = true },
            .min => .{ .min_bit = true },
            .max => .{ .max_bit = true },
        },
        .resolve_image_view = if (options.resolve_view) |rv|
            rv.asBackendType()
        else
            .null_handle,
        .resolve_image_layout = if (options.resolve_view != null)
            .attachment_optimal
        else
            .undefined,
        .load_op = switch (options.load_op) {
            .clear_color, .clear_depth_stencil => .clear,
            .load => .load,
            .dont_care => .dont_care,
        },
        .store_op = switch (options.store_op) {
            .store => .store,
            .dont_care => .dont_care,
            .none => .none,
        },
        .clear_value = switch (options.load_op) {
            .clear_color => |color| .{ .color = .{ .float_32 = color } },
            .clear_depth_stencil => |ds| .{ .depth_stencil = .{
                .depth = ds.depth,
                .stencil = ds.stencil,
            } },
            else => .{ .color = .{ .float_32 = .{ 0.0, 0.0, 0.0, 0.0 } } },
        },
    };
}

pub fn cmdBufBeginRendering(
    self: *Gx,
    cb: gpu.CmdBuf,
    options: gpu.CmdBuf.BeginRenderingOptions,
) void {
    const arena = self.arena.begin() catch @panic("OOM");
    defer self.arena.end();

    const color_attachments = arena.alloc(
        vk.RenderingAttachmentInfo,
        options.color_attachments.len,
    ) catch @panic("OOM");
    for (color_attachments, options.color_attachments) |*vk_attachment, gpu_attachment| {
        vk_attachment.* = attachmentToVk(gpu_attachment);
    }

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
        .p_depth_attachment = b: {
            const vk_depth_attachment = options.depth_attachment orelse break :b null;
            const gpu_depth_attachment = arena.create(vk.RenderingAttachmentInfo) catch @panic("OOM");
            gpu_depth_attachment.* = attachmentToVk(vk_depth_attachment);
            break :b gpu_depth_attachment;
        },
        .p_stencil_attachment = b: {
            const vk_stencil_attachment = options.stencil_attachment orelse break :b null;
            const gpu_stencil_attachment = arena.create(vk.RenderingAttachmentInfo) catch @panic("OOM");
            gpu_stencil_attachment.* = attachmentToVk(vk_stencil_attachment);
            break :b gpu_stencil_attachment;
        },
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
        shaderStagesToVk(options.stages),
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
    groups: gpu.Extent3D,
) void {
    self.backend.device.cmdDispatch(cb.asBackendType(), groups.width, groups.height, groups.depth);
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

fn bindPointToVk(self: gpu.BindPoint) vk.PipelineBindPoint {
    return switch (self) {
        .graphics => .graphics,
        .compute => .compute,
    };
}

pub fn cmdBufBindPipeline(self: *Gx, cb: gpu.CmdBuf, options: gpu.CmdBuf.BindPipelineOptions) void {
    self.backend.device.cmdBindPipeline(
        cb.asBackendType(),
        bindPointToVk(options.bind_point),
        options.pipeline.asBackendType(),
    );
}

pub fn cmdBufBindDescSet(
    self: *Gx,
    cb: gpu.CmdBuf,
    options: gpu.CmdBuf.BindDescSetOptions,
) void {
    // If this assertion fails, we need to update the code below
    comptime assert(std.meta.fields(gpu.BindPoints).len == 2);

    // Once we adopt Vulkan 1.4, this can be done in a single API call. We probably shouldn't adopt
    // the per-stage binding however unless DX12 supports it.
    {
        if (options.bind_points.graphics) {
            self.backend.device.cmdBindDescriptorSets(
                cb.asBackendType(),
                .graphics,
                options.layout.asBackendType(),
                0,
                1,
                &.{options.set.asBackendType()},
                0,
                &[0]u32{},
            );
        }

        if (options.bind_points.compute) {
            self.backend.device.cmdBindDescriptorSets(
                cb.asBackendType(),
                .compute,
                options.layout.asBackendType(),
                0,
                1,
                &.{options.set.asBackendType()},
                0,
                &[0]u32{},
            );
        }
    }
}

pub fn cmdBufEnd(self: *Gx, cb: gpu.CmdBuf) void {
    cb.endZone(self);
    self.backend.device.endCommandBuffer(cb.asBackendType()) catch |err| @panic(@errorName(err));
}

pub fn submit(self: *Gx, cbs: []const gpu.CmdBuf) void {
    const queue_submit_zone = Zone.begin(.{ .name = "queue submit", .src = @src() });
    defer queue_submit_zone.end();
    const submit_infos = [_]vk.SubmitInfo{.{
        .wait_semaphore_count = 0,
        .p_wait_semaphores = &.{},
        .p_wait_dst_stage_mask = &.{},
        .command_buffer_count = @intCast(cbs.len),
        .p_command_buffers = @ptrCast(cbs),
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
    const arena = self.arena.begin() catch @panic("OOM");
    defer self.arena.end();

    // Create the descriptor pool
    const desc_pool = b: {
        // Calculate the size of the pool
        var samplers: u32 = 0;
        var sampled_images: u32 = 0;
        var storage_images: u32 = 0;
        var uniform_buffers: u32 = 0;
        var storage_buffers: u32 = 0;

        for (options.cmds) |cmd| {
            for (cmd.layout_options.descs) |desc| {
                switch (desc.kind) {
                    .sampler => samplers += desc.count,
                    .sampled_image => sampled_images += desc.count,
                    .storage_image => storage_images += desc.count,
                    .uniform_buffer => uniform_buffers += desc.count,
                    .storage_buffer => storage_buffers += desc.count,
                }
            }
        }

        // Descriptor count must be greater than zero, so skip any that are zero
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorPoolSize.html
        var sizes = std.ArrayListUnmanaged(vk.DescriptorPoolSize)
            .initCapacity(arena, 4) catch @panic("OOM");
        if (samplers > 0) sizes.appendAssumeCapacity(.{
            .type = .sampler,
            .descriptor_count = samplers,
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
            .pool_size_count = @intCast(sizes.items.len),
            .p_pool_sizes = sizes.items.ptr,
            .flags = .{},
            .max_sets = @intCast(options.cmds.len),
        }, null) catch |err| @panic(@errorName(err));
        setName(self.backend.debug_messenger, self.backend.device, desc_pool, options.name);

        break :b desc_pool;
    };

    // Create the descriptor sets
    {
        // Collect the arguments for descriptor set creation
        const layouts = arena.alloc(vk.DescriptorSetLayout, options.cmds.len) catch @panic("OOM");
        for (layouts, options.cmds) |*layout, cmd| {
            layout.* = cmd.layout.asBackendType();
        }

        // Allocate the descriptor sets
        const results = arena.alloc(vk.DescriptorSet, options.cmds.len) catch @panic("OOM");
        self.backend.device.allocateDescriptorSets(&.{
            .descriptor_pool = desc_pool,
            .descriptor_set_count = @intCast(layouts.len),
            .p_set_layouts = layouts.ptr,
        }, results.ptr) catch |err| @panic(@errorName(err));

        // Write the results
        for (options.cmds, results) |cmd, result| {
            cmd.result.* = .fromBackendType(result);
            setName(self.backend.debug_messenger, self.backend.device, result, cmd.name);
        }
    }

    // Return the descriptor pool
    return .fromBackendType(desc_pool);
}

/// Rational for the auto batching can be found in `Gx.updateDescSets`.
pub fn descSetsUpdate(self: *Gx, updates: []const gpu.DescSet.Update) void {
    const arena = self.arena.begin() catch @panic("OOM");
    defer self.arena.end();

    var write_sets = std.ArrayListUnmanaged(vk.WriteDescriptorSet)
        .initCapacity(arena, updates.len) catch @panic("OOM");

    // Iterate over the updates
    var i: u32 = 0;
    while (i < updates.len) {
        var buffer_infos: std.ArrayListUnmanaged(vk.DescriptorBufferInfo) = .{};
        var image_infos: std.ArrayListUnmanaged(vk.DescriptorImageInfo) = .{};

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
                .sampler => |sampler| image_infos.append(arena, .{
                    .sampler = sampler.asBackendType(),
                    .image_view = .null_handle,
                    .image_layout = .undefined,
                }) catch @panic("OOM"),
                .sampled_image => |view| image_infos.append(arena, .{
                    .sampler = .null_handle,
                    .image_view = view.asBackendType(),
                    .image_layout = .read_only_optimal,
                }) catch @panic("OOM"),
                .storage_image => |view| image_infos.append(arena, .{
                    .sampler = .null_handle,
                    .image_view = view.asBackendType(),
                    .image_layout = .general,
                }) catch @panic("OOM"),
                .uniform_buf => |view| buffer_infos.append(arena, .{
                    .buffer = view.handle.asBackendType(),
                    .offset = view.offset,
                    .range = view.len,
                }) catch @panic("OOM"),
                .storage_buf => |view| buffer_infos.append(arena, .{
                    .buffer = view.handle.asBackendType(),
                    .offset = view.offset,
                    .range = view.len,
                }) catch @panic("OOM"),
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
                const buf = image_infos.toOwnedSlice(arena) catch @panic("OOM");
                assert(buf.len == batch_size);
                write_sets.appendAssumeCapacity(.{
                    .dst_set = batch_set.asBackendType(),
                    .dst_binding = batch_binding,
                    .dst_array_element = batch_index_start,
                    .descriptor_type = .sampler,
                    .descriptor_count = @intCast(buf.len),
                    .p_buffer_info = &[0]vk.DescriptorBufferInfo{},
                    .p_image_info = buf.ptr,
                    .p_texel_buffer_view = &[0]vk.BufferView{},
                });
            },
            .sampled_image => {
                const buf = image_infos.toOwnedSlice(arena) catch @panic("OOM");
                assert(buf.len == batch_size);
                write_sets.appendAssumeCapacity(.{
                    .dst_set = batch_set.asBackendType(),
                    .dst_binding = batch_binding,
                    .dst_array_element = batch_index_start,
                    .descriptor_type = .sampled_image,
                    .descriptor_count = @intCast(buf.len),
                    .p_buffer_info = &[0]vk.DescriptorBufferInfo{},
                    .p_image_info = buf.ptr,
                    .p_texel_buffer_view = &[0]vk.BufferView{},
                });
            },
            .storage_image => {
                const buf = image_infos.toOwnedSlice(arena) catch @panic("OOM");
                assert(buf.len == batch_size);
                write_sets.appendAssumeCapacity(.{
                    .dst_set = batch_set.asBackendType(),
                    .dst_binding = batch_binding,
                    .dst_array_element = batch_index_start,
                    .descriptor_type = .storage_image,
                    .descriptor_count = @intCast(buf.len),
                    .p_buffer_info = &[0]vk.DescriptorBufferInfo{},
                    .p_image_info = buf.ptr,
                    .p_texel_buffer_view = &[0]vk.BufferView{},
                });
            },
            .uniform_buf => {
                const buf = buffer_infos.toOwnedSlice(arena) catch @panic("OOM");
                assert(buf.len == batch_size);
                write_sets.appendAssumeCapacity(.{
                    .dst_set = batch_set.asBackendType(),
                    .dst_binding = batch_binding,
                    .dst_array_element = batch_index_start,
                    .descriptor_type = .uniform_buffer,
                    .descriptor_count = @intCast(buf.len),
                    .p_buffer_info = buf.ptr,
                    .p_image_info = &[0]vk.DescriptorImageInfo{},
                    .p_texel_buffer_view = &[0]vk.BufferView{},
                });
            },
            .storage_buf => {
                const buf = buffer_infos.toOwnedSlice(arena) catch @panic("OOM");
                assert(buf.len == batch_size);
                write_sets.appendAssumeCapacity(.{
                    .dst_set = batch_set.asBackendType(),
                    .dst_binding = batch_binding,
                    .dst_array_element = batch_index_start,
                    .descriptor_type = .storage_buffer,
                    .descriptor_count = @intCast(buf.len),
                    .p_buffer_info = buf.ptr,
                    .p_image_info = &[0]vk.DescriptorImageInfo{},
                    .p_texel_buffer_view = &[0]vk.BufferView{},
                });
            },
        }

        i += batch_size;
    }

    self.backend.device.updateDescriptorSets(
        @intCast(write_sets.items.len),
        write_sets.items.ptr,
        0,
        null,
    );
}

pub fn beginFrame(self: *Gx) void {
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
    if (self.validation.gte(.fast)) {
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
        .samples = samplesToVk(options.samples),
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
    // this for us. Note that many drivers incorrectly return success from here.
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
    ) catch |err| std.debug.panic("{}: {}", .{ err, options });

    // Get the memory requirements
    var dedicated_reqs: vk.MemoryDedicatedRequirements = .{
        .prefers_dedicated_allocation = vk.FALSE,
        .requires_dedicated_allocation = vk.FALSE,
    };
    const invalid_reqs: vk.MemoryRequirements = .{
        .size = std.math.maxInt(vk.DeviceSize),
        .alignment = std.math.maxInt(vk.DeviceSize),
        .memory_type_bits = 0,
    };
    var reqs2: vk.MemoryRequirements2 = .{
        .memory_requirements = invalid_reqs,
        .p_next = &dedicated_reqs,
    };
    self.backend.device.getDeviceImageMemoryRequirements(&.{
        .p_create_info = &options_vk,
        .plane_aspect = .{},
    }, &reqs2);
    const reqs = reqs2.memory_requirements;

    // Some drivers will lie and claim to support images they don't, and then leave the requirements
    // uninitialized. As a fallback also check that we actually wrote to reqs.
    if (reqs.size == invalid_reqs.size or
        reqs.alignment == invalid_reqs.alignment or
        reqs.memory_type_bits == invalid_reqs.memory_type_bits)
    {
        std.debug.panic("image format unsupported: {}", .{options});
    }

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
    self.backend.device.destroyPipeline(pipeline.asBackendType(), null);
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

fn samplesToVk(self: gpu.Samples) vk.SampleCountFlags {
    return switch (self) {
        .@"1" => .{ .@"1_bit" = true },
        .@"2" => .{ .@"2_bit" = true },
        .@"4" => .{ .@"4_bit" = true },
        .@"8" => .{ .@"8_bit" = true },
        .@"16" => .{ .@"16_bit" = true },
    };
}

fn blendFactorToVk(self: gpu.Pipeline.InitGraphicsCmd.AttachmentBlendState.Factor) vk.BlendFactor {
    return switch (self) {
        .zero => .zero,
        .one => .one,
        .src_color => .src_color,
        .one_minus_src_color => .one_minus_src_color,
        .dst_color => .dst_color,
        .one_minus_dst_color => .one_minus_dst_color,
        .src_alpha => .src_alpha,
        .one_minus_src_alpha => .one_minus_src_alpha,
        .dst_alpha => .dst_alpha,
        .one_minus_dst_alpha => .one_minus_dst_alpha,
        .constant_color => .constant_color,
        .one_minus_constant_color => .one_minus_constant_color,
        .constant_alpha => .constant_alpha,
        .one_minus_constant_alpha => .one_minus_constant_alpha,
        .src_alpha_saturate => .src_alpha_saturate,
    };
}

fn blendOpToVk(self: gpu.Pipeline.InitGraphicsCmd.AttachmentBlendState.Op) vk.BlendOp {
    return switch (self) {
        .add => .add,
        .subtract => .subtract,
        .reverse_subtract => .reverse_subtract,
        .min => .min,
        .max => .max,
    };
}

fn compareOpToVk(self: gpu.CompareOp) vk.CompareOp {
    return switch (self) {
        .never => .never,
        .lt => .less,
        .eql => .equal,
        .lte => .less_or_equal,
        .gt => .greater,
        .ne => .not_equal,
        .gte => .greater_or_equal,
        .always => .always,
    };
}

fn stencilOpToVk(self: gpu.Pipeline.InitGraphicsCmd.StencilState.OpState.Op) vk.StencilOp {
    return switch (self) {
        .keep => .keep,
        .zero => .zero,
        .replace => .replace,
        .increment_clamp => .increment_and_clamp,
        .decrement_clamp => .decrement_and_clamp,
        .invert => .invert,
        .increment_wrap => .increment_and_wrap,
        .decrement_wrap => .decrement_and_wrap,
    };
}

fn stencilOpStateToVk(self: gpu.Pipeline.InitGraphicsCmd.StencilState.OpState) vk.StencilOpState {
    return .{
        .fail_op = stencilOpToVk(self.fail_op),
        .pass_op = stencilOpToVk(self.pass_op),
        .depth_fail_op = stencilOpToVk(self.depth_fail_op),
        .compare_op = compareOpToVk(self.compare_op),
        .compare_mask = self.compare_mask,
        .write_mask = self.write_mask,
        .reference = self.reference,
    };
}

pub fn pipelinesCreateGraphics(self: *Gx, cmds: []const gpu.Pipeline.InitGraphicsCmd) void {
    const arena = self.arena.begin() catch @panic("OOM");
    defer self.arena.end();

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

    // Pipeline create info
    var pipeline_infos = std.ArrayListUnmanaged(vk.GraphicsPipelineCreateInfo)
        .initCapacity(arena, cmds.len) catch @panic("OOM");
    for (cmds) |cmd| {
        const input_assembly = arena.create(vk.PipelineInputAssemblyStateCreateInfo) catch @panic("OOM");
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

        const max_shader_stages = std.meta.fields(gpu.Pipeline.InitGraphicsCmd.Stages).len;
        var shader_stages = std.ArrayListUnmanaged(vk.PipelineShaderStageCreateInfo)
            .initCapacity(arena, max_shader_stages) catch @panic("OOM");
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

        const color_attachment_formats = gpu.ImageFormat.asBackendSlice(cmd.color_attachment_formats);
        const rendering_info = arena.create(vk.PipelineRenderingCreateInfo) catch @panic("OOM");
        rendering_info.* = .{
            .view_mask = 0,
            .color_attachment_count = @intCast(color_attachment_formats.len),
            .p_color_attachment_formats = color_attachment_formats.ptr,
            .depth_attachment_format = cmd.depth_attachment_format.asBackendType(),
            .stencil_attachment_format = cmd.stencil_attachment_format.asBackendType(),
        };

        const multisampling_info =
            arena.create(vk.PipelineMultisampleStateCreateInfo) catch @panic("OOM");
        multisampling_info.* = .{
            .sample_shading_enable = vk.FALSE,
            .rasterization_samples = samplesToVk(cmd.rasterization_samples),
            .min_sample_shading = 1.0,
            .alpha_to_coverage_enable = if (cmd.alpha_to_coverage) vk.TRUE else vk.FALSE,
            .alpha_to_one_enable = vk.FALSE,
        };

        const blend_attachment_state =
            arena.create(vk.PipelineColorBlendAttachmentState) catch @panic("OOM");
        const color_write_mask: vk.ColorComponentFlags = .{
            .r_bit = cmd.color_write_mask.r,
            .g_bit = cmd.color_write_mask.g,
            .b_bit = cmd.color_write_mask.b,
            .a_bit = cmd.color_write_mask.a,
        };
        blend_attachment_state.* = if (cmd.blend_state) |blend_state| .{
            .color_write_mask = color_write_mask,
            .blend_enable = vk.TRUE,
            .src_color_blend_factor = blendFactorToVk(blend_state.src_color_factor),
            .dst_color_blend_factor = blendFactorToVk(blend_state.dst_color_factor),
            .color_blend_op = blendOpToVk(blend_state.color_op),
            .src_alpha_blend_factor = blendFactorToVk(blend_state.src_alpha_factor),
            .dst_alpha_blend_factor = blendFactorToVk(blend_state.dst_alpha_factor),
            .alpha_blend_op = blendOpToVk(blend_state.alpha_op),
        } else .{
            .color_write_mask = color_write_mask,
            .blend_enable = vk.FALSE,
            .src_color_blend_factor = .one,
            .dst_color_blend_factor = .zero,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
        };
        const blend_state_info = arena.create(vk.PipelineColorBlendStateCreateInfo) catch @panic("OOM");
        blend_state_info.* = .{
            .logic_op_enable = vk.FALSE,
            .logic_op = .clear,
            .attachment_count = 1,
            .p_attachments = blend_attachment_state[0..1],
            .blend_constants = cmd.blend_constants,
        };
        if (cmd.logic_op) |op| {
            blend_state_info.logic_op_enable = vk.TRUE;
            blend_state_info.logic_op = switch (op) {
                .clear => .clear,
                .@"and" => .@"and",
                .and_reverse => .and_reverse,
                .copy => .copy,
                .and_inverted => .and_inverted,
                .no_op => .no_op,
                .xor => .xor,
                .@"or" => .@"or",
                .nor => .nor,
                .equivalent => .equivalent,
                .invert => .invert,
                .or_reverse => .or_reverse,
                .copy_inverted => .copy_inverted,
                .or_inverted => .or_inverted,
                .nand => .nand,
                .set => .set,
            };
        }

        var depth_stencil_state: ?*vk.PipelineDepthStencilStateCreateInfo = null;
        if (cmd.depth_state != null or cmd.stencil_state != null) {
            depth_stencil_state =
                arena.create(vk.PipelineDepthStencilStateCreateInfo) catch @panic("OOM");
            depth_stencil_state.?.* = .{
                .flags = .{},
                .depth_test_enable = vk.FALSE,
                .depth_write_enable = vk.FALSE,
                .depth_compare_op = .never,
                .depth_bounds_test_enable = vk.FALSE,
                .stencil_test_enable = vk.FALSE,
                .front = .{
                    .fail_op = .keep,
                    .pass_op = .keep,
                    .depth_fail_op = .keep,
                    .compare_op = .never,
                    .compare_mask = 0,
                    .write_mask = 0,
                    .reference = 0,
                },
                .back = .{
                    .fail_op = .keep,
                    .pass_op = .keep,
                    .depth_fail_op = .keep,
                    .compare_op = .never,
                    .compare_mask = 0,
                    .write_mask = 0,
                    .reference = 0,
                },
                .min_depth_bounds = 0.0,
                .max_depth_bounds = 0.0,
            };
            if (cmd.depth_state) |s| {
                depth_stencil_state.?.*.depth_test_enable = vk.TRUE;
                depth_stencil_state.?.*.depth_write_enable = if (s.write) vk.TRUE else vk.FALSE;
                depth_stencil_state.?.*.depth_compare_op = compareOpToVk(s.compare_op);
            }
            if (cmd.stencil_state) |s| {
                depth_stencil_state.?.*.stencil_test_enable = vk.TRUE;
                depth_stencil_state.?.*.front = stencilOpStateToVk(s.front);
                depth_stencil_state.?.*.back = stencilOpStateToVk(s.back);
            }
        }

        pipeline_infos.appendAssumeCapacity(.{
            .flags = .{},
            .stage_count = @intCast(shader_stages.items.len),
            .p_stages = shader_stages.items.ptr,
            .p_vertex_input_state = &vertex_input,
            .p_input_assembly_state = input_assembly,
            .p_viewport_state = &viewport_state,
            .p_rasterization_state = &rasterizer,
            .p_multisample_state = multisampling_info,
            .p_depth_stencil_state = depth_stencil_state,
            .p_color_blend_state = blend_state_info,
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
    const pipelines = arena.alloc(vk.Pipeline, cmds.len) catch @panic("OOM");
    const create_result = self.backend.device.createGraphicsPipelines(
        self.backend.pipeline_cache,
        @intCast(pipeline_infos.items.len),
        pipeline_infos.items.ptr,
        null,
        pipelines.ptr,
    ) catch |err| @panic(@errorName(err));
    switch (create_result) {
        .success => {},
        else => |err| @panic(@tagName(err)),
    }
    for (pipelines, cmds) |pipeline, cmd| {
        setName(self.backend.debug_messenger, self.backend.device, pipeline, cmd.name);
        cmd.result.* = .fromBackendType(pipeline);
    }
}

pub fn pipelinesCreateCompute(self: *Gx, cmds: []const gpu.Pipeline.InitComputeCmd) void {
    const arena = self.arena.begin() catch @panic("OOM");
    defer self.arena.end();
    var pipeline_infos = std.ArrayListUnmanaged(vk.ComputePipelineCreateInfo)
        .initCapacity(arena, cmds.len) catch @panic("OOM");
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
    const pipelines = arena.alloc(vk.Pipeline, cmds.len) catch @panic("OOM");
    const create_result = self.backend.device.createComputePipelines(
        self.backend.pipeline_cache,
        @intCast(pipeline_infos.items.len),
        pipeline_infos.items.ptr,
        null,
        pipelines.ptr,
    ) catch |err| @panic(@errorName(err));
    switch (create_result) {
        .success => {},
        else => |err| @panic(@tagName(err)),
    }
    for (pipelines, cmds) |pipeline, cmd| {
        setName(self.backend.debug_messenger, self.backend.device, pipeline, cmd.name);
        cmd.result.* = .fromBackendType(pipeline);
    }
}

pub fn endFrame(self: *Gx, options: Gx.EndFrameOptions) void {
    // Check if we're presenting an image this frame
    const present = options.present orelse {
        // If not, just wrap up the command pool by signaling the fence for this frame and then
        // early out.
        const end_zone = Zone.begin(.{ .name = "end", .src = @src() });
        defer end_zone.end();
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
    };

    // Acquire the next swapchain image
    const image_index = b: {
        const acquire_zone = Zone.begin(.{ .name = "acquire", .src = @src() });
        defer acquire_zone.end();

        // Mark the swapchain as dirty if the extent has changed. Many platforms will consider this
        // sub-optimal, but this isn't guaranteed. Wayland for example appears to never, or at least
        // rarely, report the swapchain as out of date or suboptimal.
        if (!std.meta.eql(present.surface_extent, self.backend.swapchain_extent)) {
            self.backend.recreate_swapchain = true;
        }

        // If the swapchain needs to be recreated, do so immediately. In theory in all except the
        // case where the swapchain is out of date, you could defer this work until the user
        // finishes resizing the window for a smoother experience.
        //
        // In practice, this is not viable.
        //
        // Empirically, under Wayland and under Windows drawing on a swapchain with the incorrect
        // size results in the image being stretched to fill the window without regard for aspect
        // ratio. This is almost certainly not what you want for any real application.
        //
        // You could attempt to compensate for this by adjusting your viewport, but this will break
        // X11 which empirically does *not* stretch the image, but leaves it the image at actual
        // size and pads the rest of the window with black.
        //
        // Theoretically you could use `VK_EXT_swapchain_maintenance1` to force the desired behavior
        // in these cases, but again this is not viable in practice, as only 24% of Windows devices
        // and 30% of Linux devices support it at the time of writing (https://vulkan.gpuinfo.org/)
        // despite it having been available for years.
        //
        // If one was determined to elide the recreation, they would need to adjust the final stage
        // of their renderer for each of these backends individually, since there's no one size fits
        // all situation. I would be hesitant to do such a thing, though, as it's unclear to me if
        // the observed behavior is even guaranteed.
        //
        // Instead, we just recreate the swapchain immediately. The only downside is a slightly
        // lower framerate during the resize than would otherwise be possible.
        if (self.backend.recreate_swapchain) {
            self.backend.setSwapchainExtent(present.surface_extent, self.hdr_metadata);
        }

        // Actually acquire the image. Drivers typically block either here or on present if the
        // image isn't yet available.
        const blocking_zone = Zone.begin(.{
            .src = @src(),
            .color = gpu.global_options.blocking_zone_color,
        });
        defer blocking_zone.end();
        while (true) {
            const acquire_result = self.backend.device.acquireNextImageKHR(
                self.backend.swapchain,
                std.math.maxInt(u64),
                self.backend.image_availables[self.frame],
                .null_handle,
            ) catch |err| switch (err) {
                error.OutOfDateKHR, error.FullScreenExclusiveModeLostEXT => {
                    self.backend.setSwapchainExtent(present.surface_extent, self.hdr_metadata);
                    continue;
                },
                error.OutOfHostMemory,
                error.OutOfDeviceMemory,
                error.Unknown,
                error.SurfaceLostKHR,
                error.DeviceLost,
                => @panic(@errorName(err)),
            };
            break :b acquire_result.image_index;
        }
    };

    const swapchain_image = self.backend.swapchain_images.items[image_index];

    // Blit the image the caller wants to present to the swapchain.
    {
        const blit_zone = Zone.begin(.{ .name = "blit", .src = @src() });
        defer blit_zone.end();

        // Create the command buffer
        const cb: gpu.CmdBuf = .init(self, .{ .src = @src(), .name = "blit to swapchain" });

        // Transition the swapchain image to transfer destination
        cb.barriers(self, .{
            .image = &.{
                .{
                    .image = .fromBackendType(swapchain_image),
                    .range = .first(.{ .color = true }),
                    .src = .{
                        .stages = .{ .top_of_pipe = true },
                        .access = .{},
                        .layout = .undefined,
                    },
                    .dst = .{
                        .stages = .{ .blit = true },
                        .access = .{ .transfer_write = true },
                        .layout = .transfer_dst,
                    },
                },
            },
        });

        // Perform the blit
        cb.blit(self, .{
            .src = .fromBackendType(present.handle.asBackendType()),
            .dst = .fromBackendType(swapchain_image),
            .regions = &.{.{
                .src = .{
                    .mip_level = 0,
                    .base_array_layer = 0,
                    .array_layers = 1,
                    .volume = .fromExtent2D(present.src_extent),
                },
                .dst = .{
                    .mip_level = 0,
                    .base_array_layer = 0,
                    .array_layers = 1,
                    .volume = .fromExtent2D(self.backend.swapchain_extent),
                },
                .aspect = .{ .color = true },
            }},
            .filter = present.filter,
        });

        // Transition the swapchain image to present source
        self.backend.device.cmdPipelineBarrier2(cb.asBackendType(), &.{
            .dependency_flags = .{},
            .memory_barrier_count = 0,
            .p_memory_barriers = &.{},
            .buffer_memory_barrier_count = 0,
            .p_buffer_memory_barriers = &.{},
            .image_memory_barrier_count = 1,
            .p_image_memory_barriers = &.{.{
                .src_stage_mask = .{ .blit_bit = true },
                .src_access_mask = .{ .transfer_write_bit = true },
                .dst_stage_mask = .{ .bottom_of_pipe_bit = true },
                .dst_access_mask = .{},
                .old_layout = .transfer_dst_optimal,
                .new_layout = .present_src_khr,
                .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                .image = swapchain_image,
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .base_mip_level = present.range.base_mip_level,
                    .level_count = present.range.mip_levels,
                    .base_array_layer = present.range.base_array_layer,
                    .layer_count = present.range.array_layers,
                },
            }},
        });

        // End the command buffer, we use our wrapper that also ends the GPU zone we created
        cmdBufEnd(self, cb);

        // Submit the command buffer, making sure to wait on the present semaphore for this
        // swapchain, image and to signal the command pool ready semaphore for this frame in flight.
        self.backend.device.queueSubmit(
            self.backend.queue,
            1,
            &.{.{
                .wait_semaphore_count = 1,
                .p_wait_semaphores = &.{self.backend.image_availables[self.frame]},
                .p_wait_dst_stage_mask = &.{.{ .top_of_pipe_bit = true }},
                .command_buffer_count = 1,
                .p_command_buffers = &.{cb.asBackendType()},
                .signal_semaphore_count = 1,
                .p_signal_semaphores = &.{self.backend.ready_for_present.items[image_index]},
                .p_next = null,
            }},
            self.backend.cmd_pool_ready[self.frame],
        ) catch |err| @panic(@errorName(err));
    }

    // Actually present the image
    {
        const queue_present_zone = Zone.begin(.{ .name = "queue present", .src = @src() });
        defer queue_present_zone.end();
        const result = self.backend.device.queuePresentKHR(
            self.backend.queue,
            &.{
                .wait_semaphore_count = 1,
                .p_wait_semaphores = &.{self.backend.ready_for_present.items[image_index]},
                .swapchain_count = 1,
                .p_swapchains = &.{self.backend.swapchain},
                .p_image_indices = &.{image_index},
                .p_results = null,
            },
        ) catch |err| b: switch (err) {
            error.OutOfDateKHR, error.FullScreenExclusiveModeLostEXT => {
                self.backend.recreate_swapchain = true;
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
            self.backend.recreate_swapchain = true;
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
        .compare_op = if (options.compare_op) |op| compareOpToVk(op) else .never,
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

    fn init(device_exts: DeviceExts, device: vk.DeviceProxy, timestamp_queries: bool) TimestampCalibration {
        if (!timestamp_queries) return .{
            .cpu = 0,
            .gpu = 0,
            .max_deviation = 0,
        };
        var calibration_results: [2]u64 = undefined;
        const getCalibratedTimestamps = device_exts.getGetCalibratedTimestampsFn().?;
        const max_deviation = getCalibratedTimestamps(
            device,
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

fn shaderStagesToVk(stages: gpu.ShaderStages) vk.ShaderStageFlags {
    comptime assert(std.meta.fields(gpu.ShaderStages).len == 3); // Update below if this fails!
    return .{
        .vertex_bit = stages.vertex,
        .fragment_bit = stages.fragment,
        .compute_bit = stages.compute,
    };
}

fn rangeToVk(range: gpu.ImageBarrier.Range) vk.ImageSubresourceRange {
    return .{
        .aspect_mask = aspectToVk(range.aspect),
        .base_mip_level = range.base_mip_level,
        .level_count = range.mip_levels,
        .base_array_layer = range.base_array_layer,
        .layer_count = range.array_layers,
    };
}

pub fn bufBarrierInit(
    options: gpu.BufBarrier.Options,
) gpu.BufBarrier {
    return .{
        .backend = .{
            .src_stage_mask = barrierStagesToVk(options.src_stages),
            .src_access_mask = accessToVk(options.src_access),
            .dst_stage_mask = barrierStagesToVk(options.dst_stages),
            .dst_access_mask = accessToVk(options.dst_access),
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

fn barrierStagesToVk(stages: gpu.BarrierStages) vk.PipelineStageFlags2 {
    comptime assert(std.meta.fields(gpu.BarrierStages).len == 10); // Update below if this fails!
    return .{
        .top_of_pipe_bit = stages.top_of_pipe,
        .vertex_shader_bit = stages.vertex,
        .early_fragment_tests_bit = stages.early_fragment_tests,
        .late_fragment_tests_bit = stages.late_fragment_tests,
        .fragment_shader_bit = stages.fragment,
        .color_attachment_output_bit = stages.color_attachment_output,
        .compute_shader_bit = stages.compute,
        .copy_bit = stages.copy,
        .blit_bit = stages.blit,
        .bottom_of_pipe_bit = stages.bottom_of_pipe,
    };
}

fn accessToVk(access: gpu.Access) vk.AccessFlags2 {
    comptime assert(std.meta.fields(gpu.Access).len == 8); // Update below if this fails!
    return .{
        .shader_read_bit = access.shader_read,
        .shader_write_bit = access.shader_write,
        .transfer_read_bit = access.transfer_read,
        .transfer_write_bit = access.transfer_write,
        .color_attachment_read_bit = access.color_attachment_read,
        .color_attachment_write_bit = access.color_attachment_write,
        .depth_stencil_attachment_read_bit = access.depth_stencil_attachment_read,
        .depth_stencil_attachment_write_bit = access.depth_stencil_attachment_write,
    };
}

fn layoutToVk(layout: gpu.ImageBarrier.Layout) vk.ImageLayout {
    return switch (layout) {
        .undefined => .undefined,
        .general => .general,
        .read_only => .read_only_optimal,
        .attachment => .attachment_optimal,
        .transfer_src => .transfer_src_optimal,
        .transfer_dst => .transfer_dst_optimal,
    };
}

pub fn cmdBufBarriers(
    self: *Gx,
    cb: gpu.CmdBuf,
    options: gpu.CmdBuf.BarriersOptions,
) void {
    const arena = self.arena.begin() catch @panic("OOM");
    defer self.arena.end();

    const image_barriers = arena.alloc(vk.ImageMemoryBarrier2, options.image.len) catch @panic("OOM");
    for (image_barriers, options.image) |*vk_image_barrier, gpu_image_barrier| {
        vk_image_barrier.* = .{
            .src_stage_mask = barrierStagesToVk(gpu_image_barrier.src.stages),
            .src_access_mask = accessToVk(gpu_image_barrier.src.access),
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .old_layout = layoutToVk(gpu_image_barrier.src.layout),
            .dst_stage_mask = barrierStagesToVk(gpu_image_barrier.dst.stages),
            .dst_access_mask = accessToVk(gpu_image_barrier.dst.access),
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .new_layout = layoutToVk(gpu_image_barrier.dst.layout),
            .image = gpu_image_barrier.image.asBackendType(),
            .subresource_range = .{
                .aspect_mask = aspectToVk(gpu_image_barrier.range.aspect),
                .base_mip_level = gpu_image_barrier.range.base_mip_level,
                .level_count = gpu_image_barrier.range.mip_levels,
                .base_array_layer = gpu_image_barrier.range.base_array_layer,
                .layer_count = gpu_image_barrier.range.array_layers,
            },
        };
    }

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

fn blitSubresourceToVk(
    self: gpu.CmdBuf.BlitOptions.Subresource,
    aspect: gpu.ImageAspect,
) vk.ImageSubresourceLayers {
    return .{
        .aspect_mask = aspectToVk(aspect),
        .mip_level = self.mip_level,
        .base_array_layer = self.base_array_layer,
        .layer_count = self.array_layers,
    };
}

fn offset3DToVk(self: gpu.Offset3D) vk.Offset3D {
    return .{ .x = self.x, .y = self.y, .z = self.z };
}

fn volumeToVk(volume: gpu.Volume) [2]vk.Offset3D {
    return .{
        offset3DToVk(volume.min),
        offset3DToVk(volume.max),
    };
}

pub fn cmdBufUploadImage(
    self: *Gx,
    cb: gpu.CmdBuf,
    dst: gpu.ImageHandle,
    src: gpu.BufHandle(.{}),
    gpu_regions: []const gpu.ImageUpload.Region,
) void {
    const arena = self.arena.begin() catch @panic("OOM");
    defer self.arena.end();

    const vk_regions = arena.alloc(vk.BufferImageCopy, gpu_regions.len) catch @panic("OOM");
    for (vk_regions, gpu_regions) |*vk_region, gpu_region| {
        vk_region.* = .{
            .buffer_offset = gpu_region.buffer_offset,
            .buffer_row_length = gpu_region.buffer_row_length orelse 0,
            .buffer_image_height = gpu_region.buffer_image_height orelse 0,
            .image_subresource = .{
                .aspect_mask = aspectToVk(gpu_region.aspect),
                .mip_level = gpu_region.mip_level,
                .base_array_layer = gpu_region.base_array_layer,
                .layer_count = gpu_region.array_layers,
            },
            .image_offset = .{
                .x = gpu_region.image_offset.x,
                .y = gpu_region.image_offset.y,
                .z = gpu_region.image_offset.z,
            },
            .image_extent = .{
                .width = gpu_region.image_extent.width,
                .height = gpu_region.image_extent.height,
                .depth = gpu_region.image_extent.depth,
            },
        };
    }

    // `cmdCopyImage` has been superseded by `cmdCopyImage2`, however there's no benefit to the
    // new API unless you need a `pNext` chain, and as such we're opting to just save the extra
    // bytes.
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
    gpu_regions: []const Gx.BufferUpload.Region,
) void {
    const arena = self.arena.begin() catch @panic("OOM");
    defer self.arena.end();

    const vk_regions = arena.alloc(vk.BufferImageCopy, gpu_regions.len) catch @panic("OOM");
    for (vk_regions, gpu_regions) |*vk_region, gpu_region| {
        vk_region.* = .{
            .src_offset = gpu_region.src_offset,
            .dst_offset = gpu_region.dst_offset,
            .size = gpu_region.size,
        };
    }

    // `cmdCopyBuffer` has been superseded by `cmdCopyBuffer2`, however there's no benefit to the
    // new API unless you need a `pNext` chain, and as such we're opting to just save the extra
    // bytes.
    self.backend.device.cmdCopyBuffer(
        cb.asBackendType(),
        src.asBackendType(),
        dst.asBackendType(),
        @intCast(vk_regions.len),
        vk_regions.ptr,
    );
}

pub fn cmdBufBlit(self: *Gx, cb: gpu.CmdBuf, options: gpu.CmdBuf.BlitOptions) void {
    var arena = self.arena.begin() catch @panic("OOM");
    defer self.arena.end();

    const regions = arena.alloc(vk.ImageBlit, options.regions.len) catch @panic("OOM");
    for (regions, options.regions) |*region_vk, region_gpu| {
        region_vk.* = .{
            .src_subresource = blitSubresourceToVk(region_gpu.src, region_gpu.aspect),
            .src_offsets = volumeToVk(region_gpu.src.volume),
            .dst_subresource = blitSubresourceToVk(region_gpu.dst, region_gpu.aspect),
            .dst_offsets = volumeToVk(region_gpu.dst.volume),
        };
    }

    self.backend.device.cmdBlitImage(
        cb.asBackendType(),
        options.src.asBackendType(),
        .transfer_src_optimal,
        options.dst.asBackendType(),
        .transfer_dst_optimal,
        @intCast(regions.len),
        regions.ptr,
        filterToVk(options.filter),
    );
}

pub fn waitIdle(self: *const Gx) void {
    self.backend.device.deviceWaitIdle() catch |err| @panic(@errorName(err));
}

fn xyColorToVk(self: gpu.XYColor) vk.XYColorEXT {
    return .{
        .x = self.x,
        .y = self.y,
    };
}

pub fn updateHdrMetadata(self: *Gx, metadata: gpu.HdrMetadata) void {
    self.backend.updateHdrMetadataImpl(metadata);
}

fn updateHdrMetadataImpl(self: *@This(), metadata: gpu.HdrMetadata) void {
    if (self.physical_device.device_exts.ext_hdr_metadata) {
        self.device.setHdrMetadataEXT(1, &.{self.swapchain}, &.{.{
            .display_primary_red = xyColorToVk(metadata.display_primary_red),
            .display_primary_green = xyColorToVk(metadata.display_primary_green),
            .display_primary_blue = xyColorToVk(metadata.display_primary_blue),
            .white_point = xyColorToVk(metadata.white_point),
            .max_luminance = metadata.max_luminance,
            .min_luminance = metadata.min_luminance,
            .max_content_light_level = metadata.max_content_light_level,
            .max_frame_average_light_level = metadata.max_frame_average_light_level,
        }});
    }
}

fn aspectToVk(self: gpu.ImageAspect) vk.ImageAspectFlags {
    return .{
        .color_bit = self.color,
        .depth_bit = self.depth,
        .stencil_bit = self.stencil,
    };
}

fn filterToVk(filter: gpu.ImageFilter) vk.Filter {
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

fn destroySwapchainViewsAndResetImages(self: *@This()) void {
    for (self.swapchain_views.items) |view| {
        self.device.destroyImageView(view, null);
    }
    self.swapchain_views.clearRetainingCapacity();
    self.swapchain_images.clearRetainingCapacity();
}

fn setSwapchainExtent(self: *@This(), extent: gpu.Extent2D, hdr_metadata: ?gpu.HdrMetadata) void {
    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    // Get the retired swapchain, if any
    const retired = self.swapchain;

    // Breaking these logs up between info and debug allows nice log viewers to dedup the first line
    // if debug logs are hidden
    if (retired != .null_handle) log.info("Recreating swapchain", .{});
    log.debug("New swap extent: {}x{}", .{ extent.width, extent.height });

    // The Vulkan spec has a bug that makes it impossible to know how long to wait before it's safe
    // to delete the old swapchain:
    //
    // There's no way to signal a fence when the present operation is done with it, and wait idle is
    // not guaranteed to wait until presentation is done. Khronos recommends using wait idle until a
    // fix is available, so that's what we do here.
    //
    // Theoretically a solution *is* now available in the form of `VK_EXT_swapchain_maintenance1`,
    // but only 24% of Windows devices and 30% of Linux devices support it at the time of writing
    // (https://vulkan.gpuinfo.org/) which leads me to believe that driver authors don't consider
    // this issue urgent, likely because in actual implementations the 'obvious' but technically
    // incorrect according to the spec solutions like using wait idle are fine.
    //
    // Reference: https://github.com/KhronosGroup/Vulkan-Docs/issues/1678
    if (retired != .null_handle) {
        self.device.deviceWaitIdle() catch |err| @panic(@errorName(err));
    }

    destroySwapchainViewsAndResetImages(self);

    const surface_capabilities = self.instance.getPhysicalDeviceSurfaceCapabilitiesKHR(
        self.physical_device.device,
        self.surface,
    ) catch |err| @panic(@errorName(err));
    self.swapchain_extent = if (surface_capabilities.current_extent.width == std.math.maxInt(u32) and
        surface_capabilities.current_extent.height == std.math.maxInt(u32))
    e: {
        break :e .{
            .width = std.math.clamp(
                extent.width,
                surface_capabilities.min_image_extent.width,
                surface_capabilities.max_image_extent.width,
            ),
            .height = std.math.clamp(
                extent.height,
                surface_capabilities.min_image_extent.height,
                surface_capabilities.max_image_extent.height,
            ),
        };
    } else .{
        .width = surface_capabilities.current_extent.width,
        .height = surface_capabilities.current_extent.height,
    };

    const max_images = if (surface_capabilities.max_image_count == 0) b: {
        break :b std.math.maxInt(u32);
    } else surface_capabilities.max_image_count;
    const min_image_count = @min(max_images, surface_capabilities.min_image_count + 1);

    var swapchain_create_info: vk.SwapchainCreateInfoKHR = .{
        .surface = self.surface,
        .min_image_count = min_image_count,
        .image_format = self.physical_device.surface_format.image_format.asBackendType(),
        .image_color_space = self.physical_device.surface_format.color_space.asBackendType(),
        .image_extent = .{
            .width = self.swapchain_extent.width,
            .height = self.swapchain_extent.height,
        },
        .image_array_layers = 1,
        .image_usage = .{
            // We're going to be blitting to the swapchain image, so we need to set the transfer bit
            .transfer_dst_bit = true,
            // One would expect that we don't need to set any other usages, however, we need to set
            // at least one of the following usage flags to be able to create an image view. Since
            // color attachment is a typical bit to set here, it seems the safest:
            //
            // https://docs.vulkan.org/spec/latest/chapters/resources.html#valid-imageview-imageusage
            .color_attachment_bit = true,
        },
        .pre_transform = surface_capabilities.current_transform,
        .composite_alpha = self.physical_device.composite_alpha,
        .present_mode = self.physical_device.present_mode,
        .clipped = vk.TRUE,
        .image_sharing_mode = .exclusive,
        .p_queue_family_indices = null,
        .queue_family_index_count = 0,
        .old_swapchain = retired,
    };

    self.swapchain = self.device.createSwapchainKHR(&swapchain_create_info, null) catch |err| @panic(@errorName(err));
    if (hdr_metadata) |some| self.updateHdrMetadataImpl(some);
    setName(self.debug_messenger, self.device, self.swapchain, .{ .str = "Main" });
    assert(self.swapchain_images.items.len == 0);
    assert(self.swapchain_views.items.len == 0);
    // It looks like we could technically just set the count to max swapchain images and not have
    // to call this twice, but this leads to best practice validation warnings so we just follow the
    // expected pattern here.
    var image_count: u32 = 0;
    const get_images_count_result = self.device.getSwapchainImagesKHR(
        self.swapchain,
        &image_count,
        null,
    ) catch |err| @panic(@errorName(err));
    if (get_images_count_result != .success) @panic(@tagName(get_images_count_result));
    if (image_count > self.swapchain_images.capacity) @panic("too many swap chain images");
    const get_images_result = self.device.getSwapchainImagesKHR(
        self.swapchain,
        &image_count,
        self.swapchain_images.items.ptr,
    ) catch |err| @panic(@errorName(err));
    self.swapchain_images.items.len = image_count;
    if (get_images_result != .success) @panic(@tagName(get_images_result));
    for (self.swapchain_images.items, 0..) |handle, i| {
        setName(self.debug_messenger, self.device, handle, .{ .str = "Swapchain", .index = i });
        const create_info: vk.ImageViewCreateInfo = .{
            .image = handle,
            .view_type = .@"2d",
            .format = self.physical_device.surface_format.image_format.asBackendType(),
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
        const view = self.device.createImageView(&create_info, null) catch |err| @panic(@errorName(err));
        setName(self.debug_messenger, self.device, view, .{ .str = "Swapchain", .index = i });
        self.swapchain_views.appendAssumeCapacity(view);
    }

    if (retired != .null_handle) {
        self.device.destroySwapchainKHR(retired, null);
    }

    self.recreate_swapchain = false;
}

const PhysicalDevice = struct {
    device: vk.PhysicalDevice = .null_handle,
    name: [vk.MAX_PHYSICAL_DEVICE_NAME_SIZE]u8 = .{0} ** vk.MAX_PHYSICAL_DEVICE_NAME_SIZE,
    index: usize = std.math.maxInt(usize),
    rank: u8 = 0,
    surface_format: gpu.SurfaceFormatQuery.Result = undefined,
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
    device_exts: DeviceExts = undefined,
};

const TimestampQueries = struct {
    pool: vk.QueryPool,
    count: u32,
    read: u32 = 0,
    write: u32 = 0,
};

const enabled_validation_features_all = [_]vk.ValidationFeatureEnableEXT{
    .gpu_assisted_ext,
    .gpu_assisted_reserve_binding_slot_ext,
    .best_practices_ext,
    .synchronization_validation_ext,
};

const enabled_validation_features_fast = [_]vk.ValidationFeatureEnableEXT{
    .best_practices_ext,
    .synchronization_validation_ext,
};

const FormatDebugMessage = struct {
    severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk.DebugUtilsMessageTypeFlagsEXT,
    data: [*c]const vk.DebugUtilsMessengerCallbackDataEXT,
    userdata: ?*anyopaque,

    pub fn format(data: FormatDebugMessage, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.writeAll("vulkan debug message:\n");

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
        }
    }
};

fn vkDebugCallback(
    severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk.DebugUtilsMessageTypeFlagsEXT,
    data: [*c]const vk.DebugUtilsMessengerCallbackDataEXT,
    userdata: ?*anyopaque,
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
        log.err("unknown severity {any}", .{severity});
        break :b .err;
    };

    // Change the severity of some messages
    if (data) |d| {
        switch (level) {
            .warn => switch (d.*.message_id_number) {
                // Ignore `BestPractices-vkCreateDevice-physical-device-features-not-retrieved`, this is
                // a false positive--we're using `vkGetPhysicalDeviceFeatures2`.
                584333584 => return vk.FALSE,
                // Ignore `BestPractices-vkBindBufferMemory-small-dedicated-allocation` and
                // `BestPractices-vkAllocateMemory-small-allocation`, our whole rendering strategy is
                // designed around this but we often have so little total memory that we trip it anyway!
                280337739, -40745094 => return vk.FALSE,
                // Don't warn us that validation is on every time validation is on, but do log it as
                // debug
                615892639, 2132353751, 1734198062, -2111305990 => level = .debug,
                // Don't warn us that the swapchain is out of date, we handle this it's not an
                // exceptional situation!
                1762589289 => level = .debug,
                // Don't warn us about skipping unsupported drivers, but do log it as debug
                0 => if (d.*.p_message_id_name) |name| {
                    if (std.mem.eql(u8, std.mem.span(name), "Loader Message")) {
                        level = .debug;
                    }
                },
                // Don't warn us about functions that return errors. This could be useful, but it leads to
                // false positives and Zig forces us to handle errors anyway. The false positive I hit is on
                // an Intel UHD GPU and occurs during device creation, presumably internally something calls
                // `vkGetPhysicalDeviceImageFormatProperties2`.
                1405170735 => level = .debug,
                else => {},
            },
            .err => switch (d.*.message_id_number) {
                // False positive `UNASSIGNED-Descriptor destroyed` when we resize the window, see
                // tracking issue:
                // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/10199
                -1575303641 => level = .debug,
                else => {},
            },
            else => {},
        }
    }

    // Otherwise log them
    const msg: FormatDebugMessage = .{
        .severity = severity,
        .message_type = message_type,
        .data = data,
        .userdata = userdata,
    };
    switch (level) {
        .err => {
            log.err("{f}", .{msg});
            @panic("validation error");
        },
        .warn => {
            log.warn("{f}", .{msg});
            return vk.FALSE;
        },
        .info => {
            log.info("{f}", .{msg});
            return vk.FALSE;
        },
        .debug => {
            log.debug("{f}", .{msg});
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

    const object_type = switch (@TypeOf(object)) {
        vk.Buffer => .buffer,
        vk.CommandBuffer => .command_buffer,
        vk.CommandPool => .command_pool,
        vk.DescriptorPool => .descriptor_pool,
        vk.DescriptorSet => .descriptor_set,
        vk.DescriptorSetLayout => .descriptor_set_layout,
        vk.DeviceMemory => .device_memory,
        vk.Fence => .fence,
        vk.Image => .image,
        vk.ImageView => .image_view,
        vk.Pipeline => .pipeline,
        vk.PipelineLayout => .pipeline_layout,
        vk.QueryPool => .query_pool,
        vk.Queue => .queue,
        vk.Sampler => .sampler,
        vk.Semaphore => .semaphore,
        vk.ShaderModule => .shader_module,
        vk.SwapchainKHR => .swapchain_khr,
        else => @compileError("unexpected type: " ++ @typeName(@TypeOf(object))),
    };

    var buf: [64:0]u8 = undefined;
    buf[buf.len] = 0;
    const name: [:0]const u8 = if (debug_name.index) |i| b: {
        break :b std.fmt.bufPrintZ(&buf, "{s} {}", .{ debug_name.str, i }) catch |err| switch (err) {
            error.NoSpaceLeft => buf[0..],
        };
    } else b: {
        break :b std.fmt.bufPrintZ(&buf, "{s}", .{debug_name.str}) catch |err| switch (err) {
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
pub const BufBarrier = vk.BufferMemoryBarrier2;
pub const ColorSpace = vk.ColorSpaceKHR;
pub const ImageFormat = vk.Format;

pub const named_color_spaces: btypes.NamedColorSpaces = .{
    .srgb_nonlinear = @intFromEnum(vk.ColorSpaceKHR.srgb_nonlinear_khr),
    .hdr10_st2084 = @intFromEnum(vk.ColorSpaceKHR.hdr10_st2084_ext),
    .bt2020_linear = @intFromEnum(vk.ColorSpaceKHR.bt2020_linear_ext),
    .hdr10_hlg = @intFromEnum(vk.ColorSpaceKHR.hdr10_hlg_ext),
    .extended_srgb_linear = @intFromEnum(vk.ColorSpaceKHR.extended_srgb_linear_ext),
    .extended_srgb_nonlinear = @intFromEnum(vk.ColorSpaceKHR.extended_srgb_nonlinear_ext),
};

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

    .r16g16b16a16_sfloat = @intFromEnum(vk.Format.r16g16b16a16_sfloat),
    .r16g16b16a16_unorm = @intFromEnum(vk.Format.r16g16b16a16_unorm),
    .r16g16b16a16_snorm = @intFromEnum(vk.Format.r16g16b16a16_snorm),

    .b5g6r5_unorm = @intFromEnum(vk.Format.b5g6r5_unorm_pack16),
    .b5g5r5a1_unorm = @intFromEnum(vk.Format.b5g5r5a1_unorm_pack16),

    .a8b8g8r8_srgb = @intFromEnum(vk.Format.a8b8g8r8_srgb_pack32),
    .a8b8g8r8_unorm = @intFromEnum(vk.Format.a8b8g8r8_unorm_pack32),
    .a8b8g8r8_snorm = @intFromEnum(vk.Format.a8b8g8r8_snorm_pack32),
    .b8g8r8a8_snorm = @intFromEnum(vk.Format.b8g8r8a8_snorm),
    .a2b10g10r10_unorm = @intFromEnum(vk.Format.a2b10g10r10_unorm_pack32),
    .a2r10g10b10_unorm = @intFromEnum(vk.Format.a2r10g10b10_unorm_pack32),
    .b10g11r11_ufloat = @intFromEnum(vk.Format.b10g11r11_ufloat_pack32),
    .r5g6b5_unorm = @intFromEnum(vk.Format.r5g6b5_unorm_pack16),
    .a1r5g5b5_unorm = @intFromEnum(vk.Format.a1r5g5b5_unorm_pack16),
    .r4g4b4a4_unorm = @intFromEnum(vk.Format.r4g4b4a4_unorm_pack16),
    .b4g4r4a4_unorm = @intFromEnum(vk.Format.b4g4r4a4_unorm_pack16),
    .r5g5b5a1_unorm = @intFromEnum(vk.Format.r5g5b5a1_unorm_pack16),
};
