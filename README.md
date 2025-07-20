# GPU

A light graphics API abstraction designed for bindless rendering on desktop and eventually console.

# Status

Work in progress, not yet usable. Check back soon.

## Goals

* Support the writing of renderers that can easily be ported to multiple desktop/console platforms
* Support the writing of renderers that can easily be migrated as the graphics landscape changes

## Non Goals

* This is not a renderer. Higher level render abstractions are not provided.
* This is not a graphics API. Some backends are stricter than others, and this library will not protect you from that.
	* You should develop against your strictest target backend with validation enabled.
* This API isn't designed for rendering approaches that involve frequent rebinding of resources.
	* These approaches to rendering are typically inefficient, and supporting them requires a much larger API surface that is beyond the scope of this library.
* Mobile/web support are not a priority.
	* There's nothing stopping you from writing a mobile or web backend, but these platforms are not factored into the API design. As such, you may have to emulate features that these platforms are late to adopt.

## Troubleshooting

LunarG provides validation layers for Vulkan which are enabled by default for debug builds. You may want to disable these in the init options if they're too slow on your setup. If you're getting crashes in the validation layers, make sure your SDK is up to date--I recommend getting them directly from LunarG as some package managers update them very infrequently.

# Backends

A Vulkan backend is provided in-repo for the time being. Vulkan was chosen because it's generally the strictest of the relevant APIs, and supports largest number of relevant platforms: Windows, Linux, and Switch.

As the library matures, the Vulkan backend will be moved into its own repo.

The ability to ship backends as separate projects is important, because some major console creators are cowards who fear open source, and as such backends for those targets can't be developed in public.

# Documentation

This library is light on documentation. You are expected to use it to access the functionality of the underlying API, documentation is typically only present where significant abstraction has been added on top of what is exposed by backend.

When terminology differs between backends, Vulkan terminology is generally favored.
