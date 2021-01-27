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

#include "RasterPipeline.h"
#include "nvvk/pipeline_vk.hpp"
#include "nvh/fileoperations.hpp"

extern std::vector<std::string> defaultSearchPaths;

RasterPipeline::RasterPipeline(
	vk::Device device,
	nvvk::DebugUtil& debug,
	vk::PipelineLayout layout,
	vk::RenderPass renderPass,
	const std::string& vtxShader,
	const std::string& fragShader,
	const std::string& debugName
)
	: m_device(device)
	, m_debug(debug)
	, m_pipelineLayout(layout)
	, m_renderPass(renderPass)
	, m_vtxShader(vtxShader)
	, m_fragShader(fragShader)
	, m_debugName(debugName)
{
	tryReload();
}

RasterPipeline::~RasterPipeline()
{
	if (m_vkPipeline)
	{
		m_device.destroyPipeline(m_vkPipeline);
	}
}

vk::Pipeline RasterPipeline::get()
{
	if (m_invalidated)
		tryReload();
	return m_vkPipeline;
}

bool RasterPipeline::tryLoadPipeline()
{
	using vkSS = vk::ShaderStageFlagBits;

	// Creating the Pipeline
	std::vector<std::string>                paths = defaultSearchPaths;
	nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_pipelineLayout, m_renderPass);
	gpb.setBlendAttachmentCount(4);
	vk::PipelineColorBlendAttachmentState blendState;
	blendState.blendEnable = false;
	blendState.colorWriteMask =
		vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
	gpb.setBlendAttachmentState(0, blendState);
	gpb.setBlendAttachmentState(1, blendState);
	gpb.setBlendAttachmentState(2, blendState);
	gpb.setBlendAttachmentState(3, blendState);
	gpb.depthStencilState.depthTestEnable = true;
	gpb.addShader(nvh::loadFile(m_vtxShader, true, paths, true), vkSS::eVertex);
	gpb.addShader(nvh::loadFile(m_fragShader, true, paths, true), vkSS::eFragment);
	gpb.addBindingDescriptions({
		{0, sizeof(nvmath::vec3)}, // Position
		{1, sizeof(nvmath::vec3)}, // Normal
		{2, sizeof(nvmath::vec3)}, // Tangent
		{3, sizeof(nvmath::vec2)}  // Texcoord0
		});
	gpb.addAttributeDescriptions({
		{0, 0, vk::Format::eR32G32B32Sfloat, 0},	// Position
		{1, 1, vk::Format::eR32G32B32Sfloat, 0},	// Normal
		{2, 2, vk::Format::eR32G32B32A32Sfloat, 0},	// Tangent
		{3, 3, vk::Format::eR32G32Sfloat, 0},		// Texcoord0
		});
	vk::Pipeline newPipeline = gpb.createPipeline();
	if (newPipeline)
	{
		m_device.waitIdle();
		m_device.destroyPipeline(m_vkPipeline);
		m_vkPipeline = newPipeline;
		m_debug.setObjectName(m_vkPipeline, m_debugName);
		return true;
	}
	return false;
}

void RasterPipeline::tryReload()
{
	if (tryLoadPipeline())
	{
		m_invalidated = false;
	}
}