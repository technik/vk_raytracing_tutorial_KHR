//----------------------------------------------------------------------------------------------------------------------
// Copyright 2021 Carmelo J Fdez-Aguera
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
#include <nvmath/nvmath.h>
#include "nvvk/allocator_vk.hpp"
#include "nvvk/appbase_vkpp.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"

// #VKRay
#include "nvvk/raytraceKHR_vk.hpp"

#include <string>
#include <vector>

class RasterPipeline
{
public:
	RasterPipeline(
		vk::Device device,
		nvvk::DebugUtil& debug,
		vk::PipelineLayout layout,
		vk::RenderPass renderPass,
		const std::string& vtxShader,
		const std::string& fragShader,
		const std::string& debugName
	);
	~RasterPipeline();

	vk::Pipeline get();
	void invalidate() { m_invalidated = true; }

private:
	bool tryLoadPipeline();
	void tryReload();

	vk::Device         m_device;
	nvvk::DebugUtil& m_debug;
	vk::PipelineLayout m_pipelineLayout;
	vk::RenderPass m_renderPass;
	vk::Pipeline	m_vkPipeline;

	std::string m_vtxShader, m_fragShader;
	std::string m_debugName;

	bool	m_invalidated{ false };
};
