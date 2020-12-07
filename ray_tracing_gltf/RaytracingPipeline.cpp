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
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/shaders_vk.hpp"

#include <vector>

extern std::vector<std::string> defaultSearchPaths;

RaytracingPipeline::RaytracingPipeline(
	vk::Device device,
	nvvk::AllocatorDedicated& alloc,
	const vk::PhysicalDeviceRayTracingPropertiesKHR& rtProperties,
	std::vector<std::string>& rayGenShaders,
	std::vector<std::string>& missShaders,
	std::vector<std::string>& closestHitShaders,
	PipelineLayoutInfo&& layoutInfo)
    : m_device(device)
    , m_alloc(alloc)
    , m_progSize(rtProperties.shaderGroupBaseAlignment)
    , m_rayGenShaders(rayGenShaders)
    , m_missShaders(missShaders)
    , m_closestHitShaders(closestHitShaders)
    , m_layoutInfo(std::forward<PipelineLayoutInfo>(layoutInfo))
{
	m_rayGenShadersOffset = 0;
	m_missShadersOffset = (uint32_t)rayGenShaders.size();
	m_hitShadersOffset    = m_missShadersOffset + (uint32_t)missShaders.size();

	const size_t numModules = rayGenShaders.size() + closestHitShaders.size() + missShaders.size();
	m_shaderPaths.reserve(numModules);
	m_shaderPaths.insert(m_shaderPaths.end(), rayGenShaders.begin(), rayGenShaders.end());
	m_shaderPaths.insert(m_shaderPaths.end(), missShaders.begin(), missShaders.end());
	m_shaderPaths.insert(m_shaderPaths.end(), closestHitShaders.begin(), closestHitShaders.end());

	// Create pipeline
	const std::vector<std::string>& paths = defaultSearchPaths;

	std::vector<vk::ShaderModule> modules;
	modules.reserve(numModules);
	for(auto& shaderPath : m_shaderPaths)
	{
		auto shaderModule = nvvk::createShaderModule(
			m_device, nvh::loadFile(shaderPath, true, paths));
		modules.push_back(shaderModule);
	}

	std::vector<vk::PipelineShaderStageCreateInfo> stages;
	stages.reserve(numModules);
	m_shaderGroups.clear();
	m_shaderGroups.reserve(numModules);

	// Ray generation group
	vk::RayTracingShaderGroupCreateInfoKHR rayGenGroup {
		vk::RayTracingShaderGroupTypeKHR::eGeneral,
		VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
		VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};

	for(uint32_t i = 0; i < m_rayGenShaders.size(); ++i)
	{
		rayGenGroup.setGeneralShader(m_rayGenShadersOffset+i);
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
		missGroup.setGeneralShader(m_missShadersOffset + i);
		stages.push_back({{}, vk::ShaderStageFlagBits::eMissKHR, modules[i+m_missShadersOffset], "main"});
		m_shaderGroups.push_back(missGroup);
	}

	// Closest hit group
	vk::RayTracingShaderGroupCreateInfoKHR closestHitGroup {
		vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
		VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
		VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};

	for(uint32_t i = 0; i < m_closestHitShaders.size(); ++i)
	{
		closestHitGroup.setClosestHitShader(m_hitShadersOffset + i);
		stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitKHR, modules[i+m_hitShadersOffset], "main"});
		m_shaderGroups.push_back(closestHitGroup);
	}

	// Pipeline layout
	vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;

	// Push constant: we want to be able to update constants used by the shaders
	vk::PushConstantRange pushConstant{
		vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eMissKHR,
		0, m_layoutInfo.pushConstantRangeSize };

	pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
	pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);

	// Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
	pipelineLayoutCreateInfo.setSetLayoutCount(static_cast<uint32_t>(m_layoutInfo.descSetLayouts.size()));
	pipelineLayoutCreateInfo.setPSetLayouts(m_layoutInfo.descSetLayouts.data());

	m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

	// Assemble the shader stages and recursion depth info into the ray tracing pipeline
	vk::RayTracingPipelineCreateInfoKHR rayPipelineInfo;
	rayPipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));  // Stages are shaders
	rayPipelineInfo.setPStages(stages.data());

	rayPipelineInfo.setGroupCount(static_cast<uint32_t>(
		m_shaderGroups.size()));
		rayPipelineInfo.setPGroups(m_shaderGroups.data());

	rayPipelineInfo.setMaxRecursionDepth(1);  // Ray depth
	rayPipelineInfo.setLayout(m_pipelineLayout);
	m_vkPipeline = static_cast<const vk::Pipeline&>(m_device.createRayTracingPipelineKHR({}, rayPipelineInfo));

	// House keeping
	for(auto& shaderModule : modules)
		m_device.destroy(shaderModule);

	// Create SBT
	auto groupCount = static_cast<uint32_t>(m_shaderGroups.size());               // 3 shaders: raygen, miss, chit
	uint32_t groupHandleSize = rtProperties.shaderGroupHandleSize;  // Size of a program identifier
	uint32_t baseAlignment   = rtProperties.shaderGroupBaseAlignment;  // Size of shader alignment

	// Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
	uint32_t sbtSize = groupCount * baseAlignment;

	std::vector<uint8_t> shaderHandleStorage(sbtSize);
	m_device.getRayTracingShaderGroupHandlesKHR(m_vkPipeline, 0, groupCount, sbtSize,
												shaderHandleStorage.data());
	// Write the handles in the SBT
	m_SBTBuffer = m_alloc.createBuffer(sbtSize, vk::BufferUsageFlagBits::eTransferSrc,
										vk::MemoryPropertyFlagBits::eHostVisible
											| vk::MemoryPropertyFlagBits::eHostCoherent);

	// Write the handles in the SBT
	void* mapped = m_alloc.map(m_SBTBuffer);
	auto* pData  = reinterpret_cast<uint8_t*>(mapped);
	for(uint32_t g = 0; g < groupCount; g++)
	{
		memcpy(pData, shaderHandleStorage.data() + g * groupHandleSize, groupHandleSize);  // raygen
		pData += baseAlignment;
	}
	m_alloc.unmap(m_SBTBuffer);


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

void RaytracingPipeline::trace(const vk::CommandBuffer& cmdBuf, const uvec3& size)
{
	// Compute offsets into the GPU sbt (could actually be done on reload
	vk::DeviceSize rayGenOffset = m_progSize * m_rayGenShadersOffset;
	vk::DeviceSize missOffset   = m_progSize * m_missShadersOffset;
	vk::DeviceSize hitOffset    = m_progSize * m_hitShadersOffset;
	vk::DeviceSize sbtSize = m_progSize * m_shaderPaths.size();

	const vk::StridedBufferRegionKHR raygenShaderBindingTable = {m_SBTBuffer.buffer, rayGenOffset, m_progSize, sbtSize};
	const vk::StridedBufferRegionKHR missShaderBindingTable = {m_SBTBuffer.buffer, missOffset, m_progSize, sbtSize};
	const vk::StridedBufferRegionKHR hitShaderBindingTable = {m_SBTBuffer.buffer, hitOffset, m_progSize, sbtSize};
	const vk::StridedBufferRegionKHR callableShaderBindingTable;

	// Dispatch the ray tracing pass
	cmdBuf.traceRaysKHR(
		&raygenShaderBindingTable,
		&missShaderBindingTable,
		&hitShaderBindingTable,
		&callableShaderBindingTable,
		size.x, size.y, size.z
		);
}

void RaytracingPipeline::tryReload()
{
	// TODO:
}