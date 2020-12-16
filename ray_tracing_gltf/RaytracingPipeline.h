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

#pragma once
#include <vulkan/vulkan.hpp>

#define NVVK_ALLOC_DEDICATED
#include "nvvk/allocator_vk.hpp"
#include "nvvk/appbase_vkpp.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"

// #VKRay
#include "nvvk/raytraceKHR_vk.hpp"

#include <string>
#include <vector>

class RaytracingPipeline
{
public:
	struct PipelineLayoutInfo
	{
		uint32_t pushConstantRangeSize;
		std::vector<vk::DescriptorSetLayout> descSetLayouts;
	};

	RaytracingPipeline(
		vk::Device device,
		nvvk::AllocatorDedicated& alloc,
		const vk::PhysicalDeviceRayTracingPropertiesKHR& rtProperties,
		std::vector<std::string>& rayGenShaders,
		std::vector<std::string>& missShaders,
		std::vector<std::string>& anyHitShaders,
		std::vector<std::string>& closestHitShaders,
		PipelineLayoutInfo&&
		);
	~RaytracingPipeline();

	void bind(const vk::CommandBuffer& cmdBuf);
	void bindDescriptorSets(
		const vk::CommandBuffer& cmdBuf,
		const vk::ArrayProxy<const vk::DescriptorSet>& descSets,
		uint32_t firstSet = 0);
	template<class T>
	void pushConstant(const vk::CommandBuffer& cmdBuf, const T&);

	void trace(const vk::CommandBuffer& cmdBuf, const uvec3& size);

	// Invalidating a pipeline will make it try to reload and recompile the shaders before binding it again
	void invalidate() { m_invalidated = true; }

private:
	bool tryLoadPipeline();
	void createSBT();
	void tryReload();

	vk::Device         m_device;
	nvvk::AllocatorDedicated& m_alloc;
	vk::PipelineLayout m_pipelineLayout;
	vk::Pipeline	m_vkPipeline;
	vk::Pipeline	m_stalePipeline;
	nvvk::Buffer	m_SBTBuffer; // Shader binding table buffer in GPU memory
	vk::DeviceSize            m_progSize;
	uint32_t m_groupHandleSize{};
	uint32_t m_sbtSize;

	uint32_t m_rayGenShadersOffset;
	uint32_t m_missShadersOffset;
	uint32_t m_cHitShadersOffset;
	uint32_t m_anyHitShadersOffset;

	std::vector<std::string> m_rayGenShaders;
	std::vector<std::string> m_missShaders;
	std::vector<std::string> m_anyHitShaders;
	std::vector<std::string> m_closestHitShaders;

	std::vector<std::string> m_shaderPaths;
	std::vector<vk::RayTracingShaderGroupCreateInfoKHR> m_shaderGroups;
	PipelineLayoutInfo m_layoutInfo;

	bool	m_invalidated {false};
};

// Inline templates
template <class T>
inline void RaytracingPipeline::pushConstant(const vk::CommandBuffer& cmdBuf, const T& pushConstant)
{
  cmdBuf.pushConstants<T>(m_pipelineLayout,
                          vk::ShaderStageFlagBits::eRaygenKHR
                              | vk::ShaderStageFlagBits::eMissKHR
                              | vk::ShaderStageFlagBits::eAnyHitKHR
                              | vk::ShaderStageFlagBits::eClosestHitKHR,
                          0, pushConstant);
}
