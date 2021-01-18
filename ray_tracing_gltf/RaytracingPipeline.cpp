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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "RaytracingPipeline.h"

#include "nvh/fileoperations.hpp"
#include "nvh/alignment.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/shaders_vk.hpp"

#include <vector>
#include <iostream>

extern std::vector<std::string> defaultSearchPaths;

RaytracingPipeline::RaytracingPipeline(
	vk::Device device,
	nvvk::AllocatorDedicated& alloc,
	const vk::PhysicalDeviceRayTracingPipelinePropertiesKHR& rtProperties,
	std::vector<std::string>& rayGenShaders,
	std::vector<std::string>& missShaders,
	std::vector<std::string>& anyHitShaders,
	std::vector<std::string>& closestHitShaders,
	PipelineLayoutInfo&& layoutInfo)
	: m_device(device)
	, m_alloc(alloc)
	, m_progSize(rtProperties.shaderGroupBaseAlignment)
	, m_rayGenShaders(rayGenShaders)
	, m_missShaders(missShaders)
	, m_anyHitShaders(anyHitShaders)
	, m_closestHitShaders(closestHitShaders)
	, m_layoutInfo(std::forward<PipelineLayoutInfo>(layoutInfo))
{
	m_rayGenShadersOffset = 0;
	m_missShadersOffset = (uint32_t)rayGenShaders.size();
	m_anyHitShadersOffset    = m_missShadersOffset + (uint32_t)missShaders.size();
	m_cHitShadersOffset    = m_anyHitShadersOffset + (uint32_t)anyHitShaders.size();

	const size_t numModules = rayGenShaders.size() + closestHitShaders.size() + missShaders.size();
	m_shaderPaths.reserve(numModules);
	m_shaderPaths.insert(m_shaderPaths.end(), rayGenShaders.begin(), rayGenShaders.end());
	m_shaderPaths.insert(m_shaderPaths.end(), missShaders.begin(), missShaders.end());
	m_shaderPaths.insert(m_shaderPaths.end(), anyHitShaders.begin(), anyHitShaders.end());
	m_shaderPaths.insert(m_shaderPaths.end(), closestHitShaders.begin(), closestHitShaders.end());

	// Pipeline layout
	vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;

	// Push constant: we want to be able to update constants used by the shaders
	vk::PushConstantRange pushConstant{
		vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR
		| vk::ShaderStageFlagBits::eMissKHR | vk::ShaderStageFlagBits::eAnyHitKHR,
		0, m_layoutInfo.pushConstantRangeSize };

	pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
	pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);

	// Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
	pipelineLayoutCreateInfo.setSetLayoutCount(static_cast<uint32_t>(m_layoutInfo.descSetLayouts.size()));
	pipelineLayoutCreateInfo.setPSetLayouts(m_layoutInfo.descSetLayouts.data());

	m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

	// Create SBT
	auto groupCount = numModules; // raygen, miss, chit
	m_groupHandleSize = rtProperties.shaderGroupHandleSize;  // Size of a program identifier
	m_groupSizeAligned =
		nvh::align_up(m_groupHandleSize, rtProperties.shaderGroupBaseAlignment);

	// Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
	m_sbtSize = groupCount * m_groupSizeAligned;

	// Allocate the SBT in GPU memory
	m_SBTBuffer = m_alloc.createBuffer(
		m_sbtSize,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddressKHR | vk::BufferUsageFlagBits::eShaderBindingTableKHR,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

	// Create pipeline
	tryReload();

	m_alloc.finalizeAndReleaseStaging();
}

RaytracingPipeline::~RaytracingPipeline()
{
	m_device.destroy(m_vkPipeline);
	m_device.destroy(m_pipelineLayout);
	m_alloc.destroy(m_SBTBuffer);
}

void RaytracingPipeline::bind(const vk::CommandBuffer& cmdBuf) 
{
	// Try reload before actually binding the pipeline
	if(m_invalidated)
		tryReload();

	cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, m_vkPipeline);
}

void RaytracingPipeline::bindDescriptorSets(
	const vk::CommandBuffer& cmdBuf,
	const vk::ArrayProxy<const vk::DescriptorSet>& descSets,
	uint32_t firstSet)
{
	cmdBuf.bindDescriptorSets(
		vk::PipelineBindPoint::eRayTracingKHR,
		m_pipelineLayout, firstSet, descSets, {});
}

void RaytracingPipeline::trace(const vk::CommandBuffer& cmdBuf, const nvmath::uvec3& size)
{
	uint32_t groupSize = m_groupSizeAligned;
	uint32_t groupStride = groupSize;


	vk::DeviceAddress sbtAddress = m_device.getBufferAddress({ m_SBTBuffer.buffer });

	using Stride = vk::StridedDeviceAddressRegionKHR;
	std::array<Stride, 4> strideAddresses{
		Stride{sbtAddress + m_rayGenShadersOffset * groupSize, groupStride, groupSize * m_rayGenShaders.size()},  // raygen
		Stride{sbtAddress + m_missShadersOffset * groupSize, groupStride, groupSize * m_missShaders.size()},  // miss
		Stride{sbtAddress + m_anyHitShadersOffset * groupSize, groupStride, groupSize * m_closestHitShaders.size()},  // hit
		Stride{0u, 0u, 0u} };                                              // callable

	cmdBuf.traceRaysKHR(
		&strideAddresses[0], &strideAddresses[1], &strideAddresses[2], &strideAddresses[3],
		size.x, size.y, size.z);
}

bool RaytracingPipeline::tryLoadPipeline()
{
	if(m_anyHitShaders.size() != m_closestHitShaders.size() && !m_anyHitShaders.empty())
	{
		std::cout << "Error: Number of closest hit and any hit shaders doesn't match\n";
		return false;
	}

	// Destroy old pipelines
	if(m_stalePipeline)
	{
		m_device.destroy(m_stalePipeline);
		m_stalePipeline = nullptr;
	}
	auto numModules = m_shaderPaths.size();
	const std::vector<std::string>& paths = defaultSearchPaths;

	std::vector<vk::ShaderModule> modules;
	modules.reserve(numModules);
	for(auto& shaderPath : m_shaderPaths)
	{
		m_device.waitIdle();
		auto shaderModule = nvvk::createShaderModule(
			m_device, nvh::loadFile(shaderPath, true, paths, true));
		if(!shaderModule)
			break;
		modules.push_back(shaderModule);
	}
	if(modules.size() < m_shaderPaths.size()) // Some module failed to compile
	{
		// Clean up
		for(auto& shaderModule : modules)
			m_device.destroy(shaderModule);
		return false;
	}

	std::vector<vk::PipelineShaderStageCreateInfo> stages;
	stages.reserve(numModules);
	m_shaderGroups.clear();
	m_shaderGroups.reserve(numModules - m_anyHitShaders.size());

	// Ray generation group
	vk::RayTracingShaderGroupCreateInfoKHR rayGenGroup {
		vk::RayTracingShaderGroupTypeKHR::eGeneral,
		VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
		VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};

	for(uint32_t i = 0; i < m_rayGenShaders.size(); ++i)
	{
		rayGenGroup.setGeneralShader(stages.size());
		stages.push_back({{}, vk::ShaderStageFlagBits::eRaygenKHR, modules[i], "main"});
		m_shaderGroups.push_back(rayGenGroup);
	}

	// Miss group
	vk::RayTracingShaderGroupCreateInfoKHR missGroup{
		vk::RayTracingShaderGroupTypeKHR::eGeneral,
		VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
		VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};

	for(uint32_t i = 0; i < m_missShaders.size(); ++i)
	{
		missGroup.setGeneralShader(stages.size());
		stages.push_back({{}, vk::ShaderStageFlagBits::eMissKHR, modules[i+m_missShadersOffset], "main"});
		m_shaderGroups.push_back(missGroup);
	}

	// Hit groups (any hit + closest hit)
	vk::RayTracingShaderGroupCreateInfoKHR hitGroup {
		vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
		VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
		VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};

	for(uint32_t i = 0; i < m_closestHitShaders.size(); ++i)
	{
		if (m_anyHitShaders.size() > 0)
		{
			hitGroup.setAnyHitShader(stages.size());
			stages.push_back({ {}, vk::ShaderStageFlagBits::eAnyHitKHR, modules[i + m_anyHitShadersOffset], "main" });
		}
		hitGroup.setClosestHitShader(stages.size());
		stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitKHR, modules[i+m_cHitShadersOffset], "main"});
		m_shaderGroups.push_back(hitGroup);
	}

	// Assemble the shader stages and recursion depth info into the ray tracing pipeline
	vk::RayTracingPipelineCreateInfoKHR rayPipelineInfo;
	rayPipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));  // Stages are shaders
	rayPipelineInfo.setPStages(stages.data());

	rayPipelineInfo.setGroupCount(static_cast<uint32_t>(
		m_shaderGroups.size()));
		rayPipelineInfo.setPGroups(m_shaderGroups.data());

	rayPipelineInfo.setMaxPipelineRayRecursionDepth(2);  // Ray depth
	rayPipelineInfo.setLayout(m_pipelineLayout);
	auto newPipeline = static_cast<const vk::Pipeline&>(
		m_device.createRayTracingPipelineKHR({}, {}, rayPipelineInfo));
	if(newPipeline)
	{
		// Stage pipeline for destruction on the next iteration.
		// Can't destroy immediately because the pipeline may be in flight in a current command buffer
		m_stalePipeline = m_vkPipeline;
		m_vkPipeline = newPipeline;
	}

	// House keeping
	for(auto& shaderModule : modules)
		m_device.destroy(shaderModule);

	return newPipeline ? true : false;
}

void RaytracingPipeline::createSBT()
{
	auto groupCount = static_cast<uint32_t>(m_shaderGroups.size());

	// Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
	std::vector<uint8_t> shaderHandleStorage(m_sbtSize);
	auto result = m_device.getRayTracingShaderGroupHandlesKHR(m_vkPipeline, 0, groupCount, m_sbtSize,
												shaderHandleStorage.data());
	if (result != vk::Result::eSuccess)
		LOGE("Fail getRayTracingShaderGroupHandlesKHR: %s", vk::to_string(result).c_str());

	// Write the handles in the SBT
	void* mapped = m_alloc.map(m_SBTBuffer);
	auto* pData  = reinterpret_cast<uint8_t*>(mapped);
	for(uint32_t g = 0; g < groupCount; g++)
	{
		memcpy(pData, shaderHandleStorage.data() + g * m_groupHandleSize, m_groupHandleSize);  // raygen
		pData += m_progSize;
	}
	m_alloc.unmap(m_SBTBuffer);
}

void RaytracingPipeline::tryReload()
{
	if(tryLoadPipeline())
	{
		createSBT();
		m_invalidated = false;
	}
}