//----------------------------------------------------	------------------------------------------------------------------
// Copyright 2020 Carmelo J Fdez-Aguera
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once
#include <vulkan/vulkan.hpp>

#define NVVK_ALLOC_DEDICATED

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "nvvk/context_vk.hpp"
#include "nvvk/allocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvmath/nvmath.h"

#include <string>

// Hold global GPU objects and properties
class RenderContext
{
public:
	static std::unique_ptr<RenderContext> create(const nvmath::vec2ui& windowSize, const std::string& windowName);
	~RenderContext();

	vk::Device					device() const { return m_vkctx.m_device; }
	vk::Instance				instance() const { return m_vkctx.m_instance; }
	vk::PhysicalDevice			physicalDevice() const { return m_vkctx.m_physicalDevice; }
	uint32_t					graphicsQueueIndex() const { return m_vkctx.m_queueGCT.familyIndex; }
	// Allocator for buffer, images, acceleration structures
	nvvk::AllocatorDedicated&	alloc() { return m_alloc; }
	nvvk::DebugUtil&			debug() { return m_debug; }

	const vk::SurfaceKHR		surface() const { return m_surface; }
	auto						window() const { return m_window; }

private:
	RenderContext(const nvmath::vec2ui& windowSize, const std::string& windowName);
	void getVkSurface(const vk::Instance& instance);

	GLFWwindow* m_window;
	nvvk::Context m_vkctx{};
	vk::SurfaceKHR m_surface;
	nvvk::AllocatorDedicated m_alloc;  // Allocator for buffer, images, acceleration structures
	nvvk::DebugUtil          m_debug;  // Utility to name objects
};
