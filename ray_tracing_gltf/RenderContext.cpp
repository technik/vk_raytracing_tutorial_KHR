//----------------------------------------------------------------------------------------------------------------------
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE

#include "RenderContext.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "nvpsystem.hpp"
#include "nvvk/context_vk.hpp"

namespace {
	// GLFW Callback functions
	static void onErrorCallback(int error, const char* description)
	{
		fprintf(stderr, "GLFW Error %d: %s\n", error, description);
	}
}

std::unique_ptr<RenderContext> RenderContext::create(
	const nvmath::vec2ui& windowSize,
	const std::string&    windowName)
{
	// Setup GLFW window
	glfwSetErrorCallback(onErrorCallback);
	if(!glfwInit())
	{
		return nullptr;
	}
	
	// Check Vulkan support
	if(!glfwVulkanSupported())
	{
		printf("GLFW: Vulkan Not Supported\n");
		return nullptr;
	}

	return std::unique_ptr<RenderContext>(new RenderContext(windowSize, windowName));
}

//----------------------------------------------------------------------------------------------------------------------
RenderContext::RenderContext(const nvmath::vec2ui& windowSize, const std::string& windowName)
{
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	m_window = glfwCreateWindow(windowSize.x, windowSize.y, windowName.c_str(), nullptr, nullptr);

	// Requesting Vulkan extensions and layers
	nvvk::ContextCreateInfo contextInfo(true);
	contextInfo.setVersion(1, 2);
	contextInfo.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);
	contextInfo.addInstanceExtension(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef WIN32
	contextInfo.addInstanceExtension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
	contextInfo.addInstanceExtension(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
	contextInfo.addInstanceExtension(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif
	contextInfo.addInstanceExtension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
	contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	contextInfo.addDeviceExtension(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
	contextInfo.addDeviceExtension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
	// #VKRay: Activate the ray tracing extension
	vk::PhysicalDeviceRayTracingFeaturesKHR raytracingFeature;
	contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_EXTENSION_NAME, false, &raytracingFeature);
	contextInfo.addDeviceExtension(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
	contextInfo.addDeviceExtension(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
	contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
	contextInfo.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
	contextInfo.addDeviceExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);

	// Creating Vulkan base application
	m_vkctx.initInstance(contextInfo);

	// Find all compatible devices
	auto compatibleDevices = m_vkctx.getCompatibleDevices(contextInfo);
	assert(!compatibleDevices.empty());
	// Use a compatible device
	m_vkctx.initDevice(compatibleDevices[0], contextInfo);

	// Window need to be opened to get the surface on which to draw
    getVkSurface(m_vkctx.m_instance);
    m_vkctx.setGCTQueueWithPresent(m_surface);

	m_alloc.init(device(), physicalDevice());
    m_debug.setup(device());
}

RenderContext::~RenderContext()
{
	if(m_surface)
	{
		vk::Instance instance = m_vkctx.m_instance;
		instance.destroySurfaceKHR(m_surface);
	}

	m_vkctx.deinit();

	glfwDestroyWindow(m_window);
	glfwTerminate();
}


void RenderContext::getVkSurface(const vk::Instance& instance)
{
	assert(instance);

	VkSurfaceKHR surface{};
	VkResult err = glfwCreateWindowSurface(instance, m_window, nullptr, &surface);
	if(err != VK_SUCCESS)
	{
		assert(!"Failed to create a Window surface");
	}

	m_surface = surface;
}