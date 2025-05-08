# GPU

A light graphics API abstraction for desktop, and eventually console. Designed for bindless rendering.

# Status

Work in progress, not yet usable. Check back soon.

# Backends

This library is just an interface. A Vulkan backend is provided in-repo for the time being for ease of development, this will eventually be moved into its own repo as the library matures.

This separation is important because I may want to develop a backend for a console whose creators are cowards that fear open source.

# Documentation

This library is light on documentation. You are expected to use it to access the functionality of the underlying API, documentation is typically only present where significant abstraction has been added on top of what is exposed by backend.

Vulkan terminology is generally favored.
