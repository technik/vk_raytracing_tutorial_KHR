/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sstream>
#include <vulkan/vulkan.hpp>

extern std::vector<std::string> defaultSearchPaths;

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION


#include "hello_vulkan.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "gltfscene.hpp"
#include "nvh/nvprint.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"

#include "nvh/alignment.hpp"
#include "shaders/binding.glsl"
#include "shaders/gltf.glsl"

#include "RaytracingPipeline.h"

#include <chrono>

#define FLAG_JITTER_AA	1
#define FLAG_DOF		2
#define FLAG_ALBEDO_85	4
#define FLAG_NO_SPEC	8
#define FLAG_NO_DIFF	16
#define FLAG_NEXT_EE	32
#define FLAG_EMIS_TRIS	64
#define FLAG_USE_ALIAS	128

// Holding the camera matrices
struct CameraMatrices
{
  nvmath::mat4f view;
  nvmath::mat4f proj;
  nvmath::mat4f viewInverse;
  // #VKRay
  nvmath::mat4f projInverse;
};

void HelloVulkan::renderUI()
{
	// Camera controls
	bool mustClean = ImGuiH::CameraWidget();

	// Path tracing controls
	ImGui::Checkbox("Accumulate", &m_accumulate);
	float logExp = log2(m_postPushC.exposure);
	ImGui::SliderFloat("EV steps", &logExp, -4.f, 6.f);
	m_postPushC.exposure = powf(2.f, logExp);
	if (ImGui::CollapsingHeader("Reference path tracer"))
	{
		mustClean |= ImGui::InputInt("Max bounces", &m_rtPushConstants.maxBounces, 1);
		mustClean |= ImGui::InputInt("First bounce", &m_rtPushConstants.firstBounce, 1);
		mustClean |= ImGui::InputInt("N paths/pixel", &m_rtPushConstants.numPathsPerPixel, 1);
		m_rtPushConstants.numPathsPerPixel = std::max(1, m_rtPushConstants.numPathsPerPixel);
		mustClean |= ImGui::InputInt("M geometry", &m_rtPushConstants.numGeomSamplesM, 1);
		mustClean |= ImGui::InputInt("M triangles", &m_rtPushConstants.numTrianglesM, 1);
		m_rtPushConstants.maxBounces = std::min(20, std::max(0, m_rtPushConstants.maxBounces));
		m_rtPushConstants.firstBounce = std::min(20, std::max(0, m_rtPushConstants.firstBounce));
		// Render flags
		bool jitterAA = renderFlag(FLAG_JITTER_AA);
		mustClean |= ImGui::Checkbox("Jitter AA", &jitterAA);
		bool dof = renderFlag(FLAG_DOF);
		mustClean |= ImGui::Checkbox("Depth of field", &dof);
		bool albedo085 = renderFlag(FLAG_ALBEDO_85);
		mustClean |= ImGui::Checkbox("Albedo 0.85", &albedo085);
		bool showSpecular = !renderFlag(FLAG_NO_SPEC);
		mustClean |= ImGui::Checkbox("Specular", &showSpecular);
		bool showDiffuse = !renderFlag(FLAG_NO_DIFF);
		mustClean |= ImGui::Checkbox("Diffuse", &showDiffuse);
		bool nextEventEstim = renderFlag(FLAG_NEXT_EE);
		mustClean |= ImGui::Checkbox("Next Event", &nextEventEstim);
		bool useEmmissiveTris = renderFlag(FLAG_EMIS_TRIS);
		bool useAliasTables = renderFlag(FLAG_USE_ALIAS);
		if (nextEventEstim)
		{
			mustClean |= ImGui::Checkbox("Emmisive Tris", &useEmmissiveTris);
			mustClean |= ImGui::Checkbox("Alias Tables", &useAliasTables);
		}

		m_rtPushConstants.renderFlags =
			(jitterAA ? FLAG_JITTER_AA : 0) |
			(dof ? FLAG_DOF : 0) |
			(albedo085 ? FLAG_ALBEDO_85 : 0) |
			(showSpecular ? 0 : FLAG_NO_SPEC) |
			(showDiffuse ? 0 : FLAG_NO_DIFF) |
			(nextEventEstim ? FLAG_NEXT_EE : 0) |
			(useEmmissiveTris ? FLAG_EMIS_TRIS : 0) |
			(useAliasTables ? FLAG_USE_ALIAS : 0);
		if (dof)
		{
			float expFocalDistance = log10f(m_rtPushConstants.focalDistance);
			mustClean |= ImGui::SliderFloat("Focal distance exp", &expFocalDistance, -5.f, 2.f);
			mustClean |= ImGui::SliderFloat("Lens radius", &m_rtPushConstants.lensRadius, 0.f, 0.5f);
			m_rtPushConstants.focalDistance = powf(10.f, expFocalDistance);
		}

		ImGui::Text("Emissive instances: %d", (int)m_emissiveInstances.size());
		ImGui::Text("Emissive triangles: %d", (int)m_emissiveTriangles.size());
	}
	if (mustClean || !m_accumulate)
	{
		resetFrame();
	}
}

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void HelloVulkan::setup(const vk::Instance&       instance,
                        const vk::Device&         device,
                        const vk::PhysicalDevice& physicalDevice,
                        uint32_t                  queueFamily)
{
  AppBase::setup(instance, device, physicalDevice, queueFamily);
  m_alloc.init(device, physicalDevice);
  m_debug.setup(m_device);
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void HelloVulkan::updateUniformBuffer(const vk::CommandBuffer& cmdBuf)
{
  // Prepare new UBO contents on host.
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);
  CameraMatrices hostUBO = {};
  hostUBO.view           = CameraManip.getMatrix();
  hostUBO.proj           = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
  // hostUBO.proj[1][1] *= -1;  // Inverting Y for Vulkan (not needed with perspectiveVK).
  hostUBO.viewInverse = nvmath::invert(hostUBO.view);
  // #VKRay
  hostUBO.projInverse = nvmath::invert(hostUBO.proj);

  // UBO on the device, and what stages access it.
  vk::Buffer deviceUBO = m_cameraMat.buffer;
  auto uboUsageStages = vk::PipelineStageFlagBits::eVertexShader
                      | vk::PipelineStageFlagBits::eRayTracingShaderKHR;

  // Ensure that the modified UBO is not visible to previous frames.
  vk::BufferMemoryBarrier beforeBarrier;
  beforeBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderRead);
  beforeBarrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
  beforeBarrier.setBuffer(deviceUBO);
  beforeBarrier.setOffset(0);
  beforeBarrier.setSize(sizeof hostUBO);
  cmdBuf.pipelineBarrier(
    uboUsageStages,
    vk::PipelineStageFlagBits::eTransfer,
    vk::DependencyFlagBits::eDeviceGroup, {}, {beforeBarrier}, {});

  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  cmdBuf.updateBuffer<CameraMatrices>(m_cameraMat.buffer, 0, hostUBO);

  // Making sure the updated UBO will be visible.
  vk::BufferMemoryBarrier afterBarrier;
  afterBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
  afterBarrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
  afterBarrier.setBuffer(deviceUBO);
  afterBarrier.setOffset(0);
  afterBarrier.setSize(sizeof hostUBO);
  cmdBuf.pipelineBarrier(
    vk::PipelineStageFlagBits::eTransfer,
    uboUsageStages,
    vk::DependencyFlagBits::eDeviceGroup, {}, {afterBarrier}, {});
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void HelloVulkan::createDescriptorSetLayout()
{
  using vkDS     = vk::DescriptorSetLayoutBinding;
  using vkDT     = vk::DescriptorType;
  using vkSS     = vk::ShaderStageFlagBits;
  uint32_t nbTxt = static_cast<uint32_t>(m_textures.size());

  auto& bind = m_descSetLayoutBind;
  // Camera matrices (binding = 0)
  bind.addBinding(vkDS(B_CAMERA, vkDT::eUniformBuffer, 1, vkSS::eVertex | vkSS::eRaygenKHR));
  bind.addBinding(vkDS(B_VERTICES, vkDT::eStorageBuffer, 1, vkSS::eRaygenKHR | vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));
  bind.addBinding(vkDS(B_INDICES, vkDT::eStorageBuffer, 1, vkSS::eRaygenKHR | vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));
  bind.addBinding(vkDS(B_NORMALS, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR));
  bind.addBinding(vkDS(B_TANGENTS, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR));
  bind.addBinding(vkDS(B_TEXCOORDS, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR));
  bind.addBinding(vkDS(B_MATERIALS, vkDT::eStorageBuffer, 1, vkSS::eFragment | vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));
  bind.addBinding(vkDS(B_MATRICES, vkDT::eStorageBuffer, 1, vkSS::eVertex | vkSS::eRaygenKHR | vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));
  auto nbTextures = static_cast<uint32_t>(m_textures.size());
  bind.addBinding(vkDS(B_TEXTURES, vkDT::eCombinedImageSampler, nbTextures, vkSS::eFragment | vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));
  bind.addBinding(vkDS(B_LIGHT_INST, vkDT::eStorageBuffer, 1, vkSS::eRaygenKHR));
  bind.addBinding(vkDS(B_LIGHT_TRIS, vkDT::eStorageBuffer, 1, vkSS::eRaygenKHR));
  bind.addBinding(vkDS(B_TRI_ALIAS, vkDT::eStorageBuffer, 1, vkSS::eRaygenKHR));
  bind.addBinding(vkDS(B_LIGHT_ALIAS, vkDT::eStorageBuffer, 1, vkSS::eRaygenKHR));


  m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
  m_descPool      = m_descSetLayoutBind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void HelloVulkan::updateDescriptorSet()
{
  std::vector<vk::WriteDescriptorSet> writes;

  // Camera matrices and scene description
  vk::DescriptorBufferInfo dbiUnif{m_cameraMat.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo vertexDesc{m_vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo indexDesc{m_indexBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo normalDesc{m_normalBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo tangentDesc{m_tangentBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo uvDesc{m_uvBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo materialDesc{m_materialBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo matrixDesc{ m_matrixBuffer.buffer, 0, VK_WHOLE_SIZE };
  vk::DescriptorBufferInfo lightInstDesc{m_lightsBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo emTrisInstDesc{m_emissiveTrianglesBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo triangleAliasDesc{ m_triangleAliasBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo instanceAliasDesc{ m_instanceAliasBuffer.buffer, 0, VK_WHOLE_SIZE};

  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_CAMERA, &dbiUnif));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_VERTICES, &vertexDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_INDICES, &indexDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_NORMALS, &normalDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_TANGENTS, &tangentDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_TEXCOORDS, &uvDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_MATERIALS, &materialDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_MATRICES, &matrixDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_LIGHT_INST, &lightInstDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_LIGHT_TRIS, &emTrisInstDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_TRI_ALIAS, &triangleAliasDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_LIGHT_ALIAS, &instanceAliasDesc));

  // All texture samplers
  std::vector<vk::DescriptorImageInfo> diit;
  for(auto& texture : m_textures)
    diit.emplace_back(texture.descriptor);
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, B_TEXTURES, diit.data()));

  // Writing the information
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Creating the pipeline layout
//
void HelloVulkan::createGraphicsPipeline()
{
  using vkSS = vk::ShaderStageFlagBits;

  vk::PushConstantRange pushConstantRanges = {vkSS::eVertex | vkSS::eFragment, 0,
                                              sizeof(ObjPushConstant)};

  // Creating the Pipeline Layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  vk::DescriptorSetLayout      descSetLayout(m_descSetLayout);
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&descSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Creating the Pipeline
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_pipelineLayout, m_offscreenRenderPass);
  gpb.depthStencilState.depthTestEnable = true;
  gpb.addShader(nvh::loadFile("shaders/vert_shader.vert.spv", true, paths, true), vkSS::eVertex);
  gpb.addShader(nvh::loadFile("shaders/frag_shader.frag.spv", true, paths, true), vkSS::eFragment);
  gpb.addBindingDescriptions(
      {{0, sizeof(nvmath::vec3)}, {1, sizeof(nvmath::vec3)}, {2, sizeof(nvmath::vec2)}});
  gpb.addAttributeDescriptions({
      {0, 0, vk::Format::eR32G32B32Sfloat, 0},  // Position
      {1, 1, vk::Format::eR32G32B32Sfloat, 0},  // Normal
      {2, 2, vk::Format::eR32G32Sfloat, 0},     // Texcoord0
  });
  m_graphicsPipeline = gpb.createPipeline();
  m_debug.setObjectName(m_graphicsPipeline, "Graphics");
}


bool isBinaryFile(const std::string& path)
{
  return path.substr(path.size() - 4) == ".glb";
}

void HelloVulkan::buildLightTables(vk::CommandBuffer cmdBuf)
{
	float totalRadiance = 0.f;
	for (size_t i = 0; i < m_gltfScene.m_nodes.size(); ++i)
	{
		auto& instance = m_gltfScene.m_nodes[i];
		auto& primitive = m_gltfScene.m_primMeshes[instance.primMesh];
		auto& material = m_gltfScene.m_materials[primitive.materialIndex];
		if (material.emissiveFactor != vec3(0.f, 0.f, 0.f))
		{
			LightInstanceInfo light;
			light.indexOffset = primitive.firstIndex;
			light.numTriangles = primitive.indexCount / 3;
			light.vtxOffset = primitive.vertexOffset;
			light.matrixIndex = i;
			light.weightedRadiance = 0.f;
			// Individual triangles
			m_emissiveTriangles.reserve(m_emissiveTriangles.size() + light.numTriangles);
			for (size_t j = 0; j < light.numTriangles; ++j)
			{
				EmissiveTrangleInfo triangle;
				triangle.vtxOffset = light.vtxOffset;
				triangle.indexOffset = light.indexOffset + 3 * j;
				triangle.matrixIndex = light.matrixIndex;
				triangle.weightedRadiance = material.emissiveFactor.norm() * triangle.area(m_gltfScene);
				m_emissiveTriangles.push_back(triangle);

				light.weightedRadiance += triangle.weightedRadiance;
			}
			totalRadiance += light.weightedRadiance;
			m_emissiveInstances.push_back(light);
		}
	}

	// Normalize radiance
	for (auto& light : m_emissiveInstances)
	{
		light.weightedRadiance /= totalRadiance;
	}
	for (auto& tri : m_emissiveTriangles)
	{
		tri.weightedRadiance /= totalRadiance;
	}

	m_rtPushConstants.numLightInstances = m_emissiveInstances.size();
	m_rtPushConstants.numEmissiveTris = m_emissiveTriangles.size();
	m_lightsBuffer = m_alloc.createBuffer(
		cmdBuf,
		m_emissiveInstances,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress);

	m_emissiveTrianglesBuffer = m_alloc.createBuffer(
		cmdBuf,
		m_emissiveTriangles,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress);

	buildTriangleAliasTable();
	buildInstanceAliasTable();

	m_triangleAliasBuffer = m_alloc.createBuffer(
		cmdBuf,
		m_triangleAliasTable,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress);

	m_instanceAliasBuffer = m_alloc.createBuffer(
		cmdBuf,
		m_instanceAliasTable,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress);
}

void HelloVulkan::buildTriangleAliasTable()
{
	const auto numBuckets = m_emissiveTriangles.size();
	float avgRadiance = 1.f / numBuckets;
	m_triangleAliasTable.reserve(numBuckets);
	std::vector<std::pair<float, size_t>> overflown;
	std::vector<std::pair<float, size_t>> empty;
	// Spread samples over initial buckets
	for (auto& tri : m_emissiveTriangles)
	{
		auto i = m_triangleAliasTable.size();
		SamplingAlias triangleAlias;
		triangleAlias.cutOff = tri.weightedRadiance * numBuckets;
		triangleAlias.Ki = i;
		m_triangleAliasTable.push_back(triangleAlias);
		if (triangleAlias.cutOff > 1)
		{
			overflown.emplace_back(triangleAlias.cutOff, i);
		}
		else
		{
			empty.emplace_back(triangleAlias.cutOff, i);
		}
	}
	// Sort buckets
	std::sort(empty.begin(), empty.end(), [](auto& a, auto& b) { return a.first < b.first; });
	std::sort(overflown.begin(), overflown.end(), [](auto& a, auto& b) { return a.first < b.first; });
	// Balance buckets and assign complementary samples
	for (int i = 0; !overflown.empty() && i < empty.size(); ++i)
	{
		auto& dst = empty[i];
		if (dst.first >= 1.f) // Not really empty
			continue;
		auto& src = overflown.back();
		float transfer = 1.f - dst.first;
		src.first -= transfer;
		// Assign complement
		m_triangleAliasTable[dst.second].Ki = src.second;
		// Move src bucket
		if (src.first <= 1.f)
		{
			empty.push_back(src);
			overflown.pop_back();
		}
	}
	for(auto& dst : empty)
	{
		// Assign final cutoffs
		m_triangleAliasTable[dst.second].cutOff = dst.first;
	}
}

void HelloVulkan::buildInstanceAliasTable()
{
	const auto numBuckets = m_emissiveInstances.size();
	float avgRadiance = 1.f / numBuckets;
	m_instanceAliasTable.reserve(numBuckets);
	std::vector<std::pair<float, size_t>> overflown;
	std::vector<std::pair<float, size_t>> empty;
	// Spread samples over initial buckets
	for (auto& light : m_emissiveInstances)
	{
		auto i = m_instanceAliasTable.size();
		SamplingAlias alias;
		alias.cutOff = light.weightedRadiance * numBuckets;
		alias.Ki = i;
		m_instanceAliasTable.push_back(alias);
		if (alias.cutOff > 1)
		{
			overflown.emplace_back(alias.cutOff, i);
		}
		else
		{
			empty.emplace_back(alias.cutOff, i);
		}
	}
	// Sort buckets
	std::sort(empty.begin(), empty.end(), [](auto& a, auto& b) { return a.first < b.first; });
	std::sort(overflown.begin(), overflown.end(), [](auto& a, auto& b) { return a.first < b.first; });
	// Balance buckets and assign complementary samples
	for (int i = 0; !overflown.empty() && i < empty.size(); ++i)
	{
		auto& dst = empty[i];
		if (dst.first >= 1.f) // Not really empty
			continue;
		auto& src = overflown.back();
		float transfer = 1.f - dst.first;
		src.first -= transfer;
		// Assign complement
		m_instanceAliasTable[dst.second].Ki = src.second;
		// Move src bucket
		if (src.first <= 1.f)
		{
			empty.push_back(src);
			overflown.pop_back();
		}
	}
	for (auto& dst : empty)
	{
		// Assign final cutoffs
		m_instanceAliasTable[dst.second].cutOff = dst.first;
	}
}

float HelloVulkan::EmissiveTrangleInfo::area(const nvh::GltfScene& scene)
{
	auto i0 = scene.m_indices[indexOffset + 0] + vtxOffset;
	auto i1 = scene.m_indices[indexOffset + 1] + vtxOffset;
	auto i2 = scene.m_indices[indexOffset + 2] + vtxOffset;
	auto mtx = scene.m_nodes[matrixIndex].worldMatrix;
	auto pos0 = vec3(mtx * vec4(scene.m_positions[i0], 1.f));
	auto pos1 = vec3(mtx * vec4(scene.m_positions[i1], 1.f));
	auto pos2 = vec3(mtx * vec4(scene.m_positions[i2], 1.f));
	auto triangleNormal = cross(pos2 - pos0, pos1 - pos0);
	return triangleNormal.norm() / 2;
}

//----------------------------------------------------------------------------------------------
std::vector<vec4f> generateTangentSpace(
	const std::vector<vec3f>& positions,
	const std::vector<vec3f>& normals,
	const std::vector<vec2f>& uvs,
	const std::vector<uint32_t>& indices,
	size_t                       nIndices,
	size_t                       nVertices,
	size_t                       indexOffset,
	size_t                       vertexOffset)
{
	// Prepare data
	std::vector<vec4f> tangentVectors;
	tangentVectors.resize(nVertices);
	memset(tangentVectors.data(), 0, nVertices * sizeof(vec4f));  // Clear the tangents

	// Accumulate per-triangle normals
	auto indexEnd = nIndices + indexOffset;
	for (int i = indexOffset; i < indexEnd; i += 3)  // Iterate over all triangles
	{
		auto i0 = indices[i + 0];
		auto i1 = indices[i + 1];
		auto i2 = indices[i + 2];

		vec2f localUvs[3] = { uvs[i0 + vertexOffset], uvs[i1 + vertexOffset], uvs[i2 + vertexOffset] };
		vec3f localPos[3] = { positions[i0 + vertexOffset], positions[i1 + vertexOffset],
								positions[i2 + vertexOffset] };

		vec2f deltaUV1 = localUvs[1] - localUvs[0];
		vec2f deltaUV2 = localUvs[2] - localUvs[0];

		vec3f deltaPos1 = localPos[1] - localPos[0];
		vec3f deltaPos2 = localPos[2] - localPos[0];

		auto determinant = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
		if (determinant == 0)
			continue; // Skip degenerated triangles

		// Unnormalized tangent
		vec3f triangleTangent = (deltaPos1 * deltaUV2.y - deltaUV1.y * deltaPos2) * (1 / determinant);
		//assert(abs(determinant) > 1e-4);

		tangentVectors[i0] +=
			vec4f(triangleTangent.x, triangleTangent.y, triangleTangent.z, determinant);
		tangentVectors[i1] +=
			vec4f(triangleTangent.x, triangleTangent.y, triangleTangent.z, determinant);
		tangentVectors[i2] +=
			vec4f(triangleTangent.x, triangleTangent.y, triangleTangent.z, determinant);

		//assert(dot(tangentVectors[i0], tangentVectors[i0]) > 0.99f);
		//assert(dot(tangentVectors[i1], tangentVectors[i1]) > 0.99f);
		//assert(dot(tangentVectors[i2], tangentVectors[i2]) > 0.99f);
	}

	// Orthonormalize per vertex
	for (int i = 0; i < tangentVectors.size(); ++i)
	{
		auto& tangent = tangentVectors[i];
		vec3f tangent3 = { tangent.x, tangent.y, tangent.z };
		auto& normal = normals[i];

		tangent3 = tangent3 - (dot(tangent3, normal) * normal);  // Orthogonal tangent
		tangent3 = normalize(tangent3);                          // Orthonormal tangent
		tangent = { tangent3.x, tangent3.y, tangent3.z, signbit(-tangent.w) ? -1.f : 1.f };
	}

	return tangentVectors;
}

//--------------------------------------------------------------------------------------------------
// Loading the OBJ file and setting up all buffers
//
void HelloVulkan::loadScene(const std::string& filename)
{
	if (filename.size() < 5) // Need a filename with extension
		return;
  using vkBU = vk::BufferUsageFlagBits;
  tinygltf::Model    tmodel;
  tinygltf::TinyGLTF tcontext;
  std::string        warn, error;

  LOGI("Loading file: %s\n", filename.c_str());
  auto timer = std::chrono::high_resolution_clock();
  auto t0    = timer.now();
  bool loadSuccess =
	  isBinaryFile(filename) ? tcontext.LoadBinaryFromFile(&tmodel, &error, &warn, filename) : tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, filename);
  if(!loadSuccess)
  {
    assert(!"Error while loading scene");
  }

  auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(timer.now() - t0);
  LOGI("Gltf Load time: %d\n", dt.count());

  LOGW(warn.c_str());
  LOGE(error.c_str());

  m_gltfScene.importMaterials(tmodel);
  m_gltfScene.importDrawableNodes(tmodel,
	  nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0 | nvh::GltfAttributes::Tangent);

  // Create the buffers on Device and copy vertices, indices and materials
  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  vk::CommandBuffer cmdBuf = cmdBufGet.createCommandBuffer();

  m_vertexBuffer =
      m_alloc.createBuffer(cmdBuf, m_gltfScene.m_positions,
                           vkBU::eVertexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress
                               | vkBU::eAccelerationStructureBuildInputReadOnlyKHR);
  m_indexBuffer =
      m_alloc.createBuffer(cmdBuf, m_gltfScene.m_indices,
                           vkBU::eIndexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress
                               | vkBU::eAccelerationStructureBuildInputReadOnlyKHR);
  m_normalBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_normals,
                                    vkBU::eVertexBuffer | vkBU::eStorageBuffer);
  /*std::vector<vec4f> tangents;
  if (m_gltfScene.m_tangents.size()
	  != m_gltfScene.m_positions.size())  // No tangents provided. Generate them
  {
	  m_gltfScene.m_tangents.clear();
	  for (const auto& primitive : m_gltfScene.m_primMeshes)
	  {
		  tangents = generateTangentSpace(m_gltfScene.m_positions, m_gltfScene.m_normals,
			  m_gltfScene.m_texcoords0, m_gltfScene.m_indices,
			  primitive.indexCount, primitive.vertexCount,
			  primitive.firstIndex, primitive.vertexOffset);
		  m_gltfScene.m_tangents.insert(m_gltfScene.m_tangents.end(), tangents.begin(), tangents.end());
	  }
  }*/

  m_tangentBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_tangents,
									vkBU::eVertexBuffer | vkBU::eStorageBuffer);
  m_uvBuffer     = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_texcoords0,
                                    vkBU::eVertexBuffer | vkBU::eStorageBuffer);

  buildLightTables(cmdBuf);

  // Copying all materials, only the elements we need
  std::vector<GltfShadeMaterial> shadeMaterials;
  for(auto& m : m_gltfScene.m_materials)
  {
	  shadeMaterials.emplace_back(
		  GltfShadeMaterial{
			m.pbrBaseColorFactor,
			m.pbrBaseColorTexture,
			m.emissiveFactor,
			m.emissiveTexture,
			m.normalTexture,
			m.pbrMetallicRoughnessTexture,
			m.pbrRoughnessFactor,
			m.pbrMetallicFactor });
  }
  m_materialBuffer = m_alloc.createBuffer(cmdBuf, shadeMaterials, vkBU::eStorageBuffer);

  // Instance Matrices used by rasterizer
  std::vector<nvmath::mat4f> nodeMatrices;
  for(auto& node : m_gltfScene.m_nodes)
  {
    nodeMatrices.emplace_back(node.worldMatrix);
  }
  m_matrixBuffer = m_alloc.createBuffer(cmdBuf, nodeMatrices, vkBU::eStorageBuffer);

  // The following is used to find the primitive mesh information in the CHIT
  std::vector<RtPrimitiveLookup> primLookup;
  for(auto& primMesh : m_gltfScene.m_primMeshes)
  {
    primLookup.push_back({primMesh.firstIndex, primMesh.vertexOffset, primMesh.materialIndex, primMesh.indexCount});
  }
  m_rtPrimLookup =
      m_alloc.createBuffer(cmdBuf, primLookup, vk::BufferUsageFlagBits::eStorageBuffer);


  // Creates all textures found
  createTextureImages(cmdBuf, tmodel);
  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();

  m_debug.setObjectName(m_vertexBuffer.buffer, "Vertex");
  m_debug.setObjectName(m_indexBuffer.buffer, "Index");
  m_debug.setObjectName(m_normalBuffer.buffer, "Normal");
  m_debug.setObjectName(m_tangentBuffer.buffer, "Tangent");
  m_debug.setObjectName(m_uvBuffer.buffer, "TexCoord");
  m_debug.setObjectName(m_materialBuffer.buffer, "Material");
  m_debug.setObjectName(m_matrixBuffer.buffer, "Matrix");

  dt = std::chrono::duration_cast<std::chrono::milliseconds>(timer.now() - t0);
  LOGI("Total Load time: %d\n", dt.count());

  if (!m_gltfScene.m_cameras.empty())
  {
	  // Load first camera by default
	  auto& sceneCam = tmodel.cameras.front();
	  CameraManip.setFov(sceneCam.perspective.yfov * 180 / 3.14159265f);
	  CameraManip.setMatrix(m_gltfScene.m_cameras.front().worldMatrix);
  }
}


//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer()
{
  using vkBU = vk::BufferUsageFlagBits;
  using vkMP = vk::MemoryPropertyFlagBits;

  m_cameraMat = m_alloc.createBuffer(sizeof(CameraMatrices),
                                     vkBU::eUniformBuffer | vkBU::eTransferDst, vkMP::eDeviceLocal);
  m_debug.setObjectName(m_cameraMat.buffer, "cameraMat");
}

//--------------------------------------------------------------------------------------------------
// Creating all textures and samplers
//
void HelloVulkan::createTextureImages(const vk::CommandBuffer& cmdBuf, tinygltf::Model& gltfModel)
{
  using vkIU = vk::ImageUsageFlagBits;

  vk::SamplerCreateInfo samplerCreateInfo{
      {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
  samplerCreateInfo.setMaxLod(FLT_MAX);
  vk::Format format = vk::Format::eR8G8B8A8Srgb;

  auto addDefaultTexture = [this]() {
    // Make dummy image(1,1), needed as we cannot have an empty array
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
    std::array<uint8_t, 4>   white = {255, 255, 255, 255};
    m_textures.emplace_back(m_alloc.createTexture(
        cmdBuf, 4, white.data(), nvvk::makeImage2DCreateInfo(vk::Extent2D{1, 1}), {}));
    m_debug.setObjectName(m_textures.back().image, "dummy");
  };

  if(gltfModel.images.empty())
  {
    addDefaultTexture();
    return;
  }

  m_textures.reserve(gltfModel.images.size());
  for(size_t i = 0; i < gltfModel.images.size(); i++)
  {
    auto&        gltfimage  = gltfModel.images[i];
    void*        buffer     = &gltfimage.image[0];
    VkDeviceSize bufferSize = gltfimage.image.size();
    auto         imgSize    = vk::Extent2D(gltfimage.width, gltfimage.height);

    if(bufferSize == 0 || gltfimage.width == -1 || gltfimage.height == -1)
    {
      addDefaultTexture();
      continue;
    }

    vk::ImageCreateInfo imageCreateInfo =
        nvvk::makeImage2DCreateInfo(imgSize, format, vkIU::eSampled, true);

    nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, buffer, imageCreateInfo);
    nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    m_textures.emplace_back(m_alloc.createTexture(image, ivInfo, samplerCreateInfo));

    m_debug.setObjectName(m_textures[i].image, std::string("Txt" + std::to_string(i)).c_str());
  }
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void HelloVulkan::destroyResources()
{
  m_device.destroy(m_graphicsPipeline);
  m_device.destroy(m_pipelineLayout);
  m_device.destroy(m_descPool);
  m_device.destroy(m_descSetLayout);
  m_alloc.destroy(m_cameraMat);

  m_alloc.destroy(m_vertexBuffer);
  m_alloc.destroy(m_normalBuffer);
  m_alloc.destroy(m_tangentBuffer);
  m_alloc.destroy(m_uvBuffer);
  m_alloc.destroy(m_indexBuffer);
  m_alloc.destroy(m_materialBuffer);
  m_alloc.destroy(m_matrixBuffer);
  m_alloc.destroy(m_rtPrimLookup);

  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }

  //#Post
  m_device.destroy(m_postPipeline);
  m_device.destroy(m_postPipelineLayout);
  m_device.destroy(m_postDescPool);
  m_device.destroy(m_postDescSetLayout);
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);
  m_device.destroy(m_offscreenRenderPass);
  m_device.destroy(m_offscreenFramebuffer);

  // #VKRay
  m_rtBuilder.destroy();
  m_device.destroy(m_rtDescPool);
  m_device.destroy(m_rtDescSetLayout);
  m_rtPipeline.reset();

  // Light sampling
  m_alloc.destroy(m_lightsBuffer);
  m_alloc.destroy(m_instanceAliasBuffer);
  m_alloc.destroy(m_triangleAliasBuffer);
  m_alloc.destroy(m_emissiveTrianglesBuffer);
}

//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode
//
void HelloVulkan::rasterize(const vk::CommandBuffer& cmdBuf)
{
  using vkPBP = vk::PipelineBindPoint;
  using vkSS  = vk::ShaderStageFlagBits;

  std::vector<vk::DeviceSize> offsets = {0, 0, 0};

  m_debug.beginLabel(cmdBuf, "Rasterize");

  // Dynamic Viewport
  cmdBuf.setViewport(0, {vk::Viewport(0, 0, (float)m_size.width, (float)m_size.height, 0, 1)});
  cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});

  // Drawing all triangles
  cmdBuf.bindPipeline(vkPBP::eGraphics, m_graphicsPipeline);
  cmdBuf.bindDescriptorSets(vkPBP::eGraphics, m_pipelineLayout, 0, {m_descSet}, {});
  std::vector<vk::Buffer> vertexBuffers = {m_vertexBuffer.buffer, m_normalBuffer.buffer,
                                           m_uvBuffer.buffer};
  cmdBuf.bindVertexBuffers(0, static_cast<uint32_t>(vertexBuffers.size()), vertexBuffers.data(),
                           offsets.data());
  cmdBuf.bindIndexBuffer(m_indexBuffer.buffer, 0, vk::IndexType::eUint32);

  uint32_t idxNode = 0;
  for(auto& node : m_gltfScene.m_nodes)
  {
    auto& primitive = m_gltfScene.m_primMeshes[node.primMesh];

    m_pushConstant.instanceId = idxNode++;
    m_pushConstant.materialId = primitive.materialIndex;
    cmdBuf.pushConstants<ObjPushConstant>(
        m_pipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
        m_pushConstant);
    cmdBuf.drawIndexed(primitive.indexCount, 1, primitive.firstIndex, primitive.vertexOffset, 0);
  }

  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::onResize(int /*w*/, int /*h*/)
{
  createOffscreenRender();
  updatePostDescriptorSet();
  updateRtDescriptorSet();
  resetFrame();
}

//////////////////////////////////////////////////////////////////////////
// Post-processing
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void HelloVulkan::createOffscreenRender()
{
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);

  // Creating the color image
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenColorFormat,
                                                       vk::ImageUsageFlagBits::eColorAttachment
                                                           | vk::ImageUsageFlagBits::eSampled
                                                           | vk::ImageUsageFlagBits::eStorage);


    nvvk::Image             image  = m_alloc.createImage(colorCreateInfo);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    m_offscreenColor               = m_alloc.createTexture(image, ivInfo, vk::SamplerCreateInfo());
    m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Creating the depth buffer
  auto depthCreateInfo =
      nvvk::makeImage2DCreateInfo(m_size, m_offscreenDepthFormat,
                                  vk::ImageUsageFlagBits::eDepthStencilAttachment);
  {
    nvvk::Image image = m_alloc.createImage(depthCreateInfo);

    vk::ImageViewCreateInfo depthStencilView;
    depthStencilView.setViewType(vk::ImageViewType::e2D);
    depthStencilView.setFormat(m_offscreenDepthFormat);
    depthStencilView.setSubresourceRange({vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1});
    depthStencilView.setImage(image.image);

    m_offscreenDepth = m_alloc.createTexture(image, depthStencilView);
  }

  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenColor.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eGeneral);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenDepth.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                vk::ImageAspectFlagBits::eDepth);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Creating a renderpass for the offscreen
  if(!m_offscreenRenderPass)
  {
    m_offscreenRenderPass =
        nvvk::createRenderPass(m_device, {m_offscreenColorFormat}, m_offscreenDepthFormat, 1, true,
                               true, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral);
  }

  // Creating the frame buffer for offscreen
  std::vector<vk::ImageView> attachments = {m_offscreenColor.descriptor.imageView,
                                            m_offscreenDepth.descriptor.imageView};

  m_device.destroy(m_offscreenFramebuffer);
  vk::FramebufferCreateInfo info;
  info.setRenderPass(m_offscreenRenderPass);
  info.setAttachmentCount(2);
  info.setPAttachments(attachments.data());
  info.setWidth(m_size.width);
  info.setHeight(m_size.height);
  info.setLayers(1);
  m_offscreenFramebuffer = m_device.createFramebuffer(info);
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void HelloVulkan::createPostPipeline()
{
  // Push constants in the fragment shader
  vk::PushConstantRange pushConstantRanges = {vk::ShaderStageFlagBits::eFragment, 0, sizeof(PostPushConstant)};

  // Creating the pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&m_postDescSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_postPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Pipeline: completely generic, no vertices
  std::vector<std::string> paths = defaultSearchPaths;

  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout,
                                                            m_renderPass);
  pipelineGenerator.addShader(nvh::loadFile("shaders/passthrough.vert.spv", true, paths, true),
                              vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("shaders/post.frag.spv", true, paths, true),
                              vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  m_postPipeline = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_postPipeline, "post");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void HelloVulkan::createPostDescriptor()
{
  using vkDS = vk::DescriptorSetLayoutBinding;
  using vkDT = vk::DescriptorType;
  using vkSS = vk::ShaderStageFlagBits;

  m_postDescSetLayoutBind.addBinding(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  m_postDescSetLayout = m_postDescSetLayoutBind.createLayout(m_device);
  m_postDescPool      = m_postDescSetLayoutBind.createPool(m_device);
  m_postDescSet       = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Update the output
//
void HelloVulkan::updatePostDescriptorSet()
{
  vk::WriteDescriptorSet writeDescriptorSets =
      m_postDescSetLayoutBind.makeWrite(m_postDescSet, 0, &m_offscreenColor.descriptor);
  m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void HelloVulkan::drawPost(vk::CommandBuffer cmdBuf)
{
  m_debug.beginLabel(cmdBuf, "Post");

  cmdBuf.setViewport(0, {vk::Viewport(0, 0, (float)m_size.width, (float)m_size.height, 0, 1)});
  cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});

  m_postPushC.aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
  cmdBuf.pushConstants<PostPushConstant>(m_postPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0,
                              m_postPushC);
  cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_postPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_postPipelineLayout, 0,
                            m_postDescSet, {});
  cmdBuf.draw(3, 1, 0, 0);

  m_debug.endLabel(cmdBuf);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Initialize Vulkan ray tracing
// #VKRay
void HelloVulkan::initRayTracing()
{
  // Requesting ray tracing properties
  auto properties =
      m_physicalDevice.getProperties2<vk::PhysicalDeviceProperties2,
                                      vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
  m_rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
  m_rtBuilder.setup(m_device, &m_alloc, m_graphicsQueueIndex);
}

//--------------------------------------------------------------------------------------------------
// Converting a GLTF primitive in the Raytracing Geometry used for the BLAS
//
nvvk::RaytracingBuilderKHR::BlasInput HelloVulkan::primitiveToGeometry(
    const nvh::GltfPrimMesh& prim)
{
  // Building part
  vk::DeviceAddress vertexAddress = m_device.getBufferAddress({m_vertexBuffer.buffer});
  vk::DeviceAddress indexAddress  = m_device.getBufferAddress({m_indexBuffer.buffer});

  vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
  triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);
  triangles.setVertexData(vertexAddress);
  triangles.setVertexStride(sizeof(nvmath::vec3f));
  triangles.setIndexType(vk::IndexType::eUint32);
  triangles.setIndexData(indexAddress);
  triangles.setTransformData({});
  triangles.setMaxVertex(prim.vertexCount);

  // Setting up the build info of the acceleration
  vk::AccelerationStructureGeometryKHR asGeom;
  asGeom.setGeometryType(vk::GeometryTypeKHR::eTriangles);
  asGeom.setFlags(vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation);  // For AnyHit
  asGeom.geometry.setTriangles(triangles);

  vk::AccelerationStructureBuildRangeInfoKHR offset;
  offset.setFirstVertex(prim.vertexOffset);
  offset.setPrimitiveCount(prim.indexCount / 3);
  offset.setPrimitiveOffset(prim.firstIndex * sizeof(uint32_t));
  offset.setTransformOffset(0);

  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);
  return input;
}

//--------------------------------------------------------------------------------------------------
//
//
void HelloVulkan::createBottomLevelAS()
{
  // BLAS - Storing each primitive in a geometry
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
  allBlas.reserve(m_gltfScene.m_primMeshes.size());
  for(auto& primMesh : m_gltfScene.m_primMeshes)
  {
    auto geo = primitiveToGeometry(primMesh);
    allBlas.push_back({geo});
  }
  m_rtBuilder.buildBlas(allBlas, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
}

void HelloVulkan::createTopLevelAS()
{
  std::vector<nvvk::RaytracingBuilderKHR::Instance> tlas;
  tlas.reserve(m_gltfScene.m_nodes.size());
  uint32_t instID = 0;
  for(auto& node : m_gltfScene.m_nodes)
  {
    nvvk::RaytracingBuilderKHR::Instance rayInst;
    rayInst.transform        = node.worldMatrix;
    rayInst.instanceCustomId = node.primMesh;  // gl_InstanceCustomIndexEXT: to find which primitive
    rayInst.blasId           = node.primMesh;
    rayInst.flags            = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    rayInst.hitGroupId       = 0;  // We will use the same hit group for all objects
    tlas.emplace_back(rayInst);
  }
  m_rtBuilder.buildTlas(tlas, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
}

//--------------------------------------------------------------------------------------------------
// This descriptor set holds the Acceleration structure and the output image
//
void HelloVulkan::createRtDescriptorSet()
{
  using vkDT   = vk::DescriptorType;
  using vkSS   = vk::ShaderStageFlagBits;
  using vkDSLB = vk::DescriptorSetLayoutBinding;

  m_rtDescSetLayoutBind.addBinding(vkDSLB(0, vkDT::eAccelerationStructureKHR, 1,
                                          vkSS::eRaygenKHR | vkSS::eClosestHitKHR));  // TLAS
  m_rtDescSetLayoutBind.addBinding(
      vkDSLB(1, vkDT::eStorageImage, 1, vkSS::eRaygenKHR));  // Output image
  m_rtDescSetLayoutBind.addBinding(vkDSLB(
      2, vkDT::eStorageBuffer, 1, vkSS::eRaygenKHR | vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));  // Primitive info

  m_rtDescPool      = m_rtDescSetLayoutBind.createPool(m_device);
  m_rtDescSetLayout = m_rtDescSetLayoutBind.createLayout(m_device);
  m_rtDescSet       = m_device.allocateDescriptorSets({m_rtDescPool, 1, &m_rtDescSetLayout})[0];

  vk::AccelerationStructureKHR                   tlas = m_rtBuilder.getAccelerationStructure();
  vk::WriteDescriptorSetAccelerationStructureKHR descASInfo;
  descASInfo.setAccelerationStructureCount(1);
  descASInfo.setPAccelerationStructures(&tlas);
  vk::DescriptorImageInfo imageInfo{
      {}, m_offscreenColor.descriptor.imageView, vk::ImageLayout::eGeneral};
  vk::DescriptorBufferInfo primitiveInfoDesc{m_rtPrimLookup.buffer, 0, VK_WHOLE_SIZE};

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 0, &descASInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 1, &imageInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 2, &primitiveInfoDesc));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void HelloVulkan::updateRtDescriptorSet()
{
  using vkDT = vk::DescriptorType;

  // (1) Output buffer
  vk::DescriptorImageInfo imageInfo{
      {}, m_offscreenColor.descriptor.imageView, vk::ImageLayout::eGeneral};
  vk::WriteDescriptorSet wds{m_rtDescSet, 1, 0, 1, vkDT::eStorageImage, &imageInfo};
  m_device.updateDescriptorSets(wds, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void HelloVulkan::createRtPipeline()
{
	std::vector<std::string> rayGenShaders = { "shaders/pathtrace.rgen.spv" };
	std::vector<std::string> missShaders = { "shaders/pathtrace.rmiss.spv", "shaders/raytraceShadow.rmiss.spv" };
	std::vector<std::string> chitShaders = { "shaders/pathtrace.rchit.spv" };
	std::vector<std::string> anyHitShaders;// = { "shaders/pathtrace.rahit.spv" };

	RaytracingPipeline::PipelineLayoutInfo pipelineLayout;
	pipelineLayout.descSetLayouts = { m_rtDescSetLayout, m_descSetLayout };
	pipelineLayout.pushConstantRangeSize = sizeof(RtPushConstant);

	m_rtPipeline = std::make_unique<RaytracingPipeline>(
		m_device, m_alloc, m_rtProperties,
		rayGenShaders, missShaders, anyHitShaders, chitShaders,
		std::move(pipelineLayout)
		);
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void HelloVulkan::raytrace(const vk::CommandBuffer& cmdBuf, const nvmath::vec4f& clearColor)
{
  updateFrame();

  m_debug.beginLabel(cmdBuf, "Ray trace");
  // Initializing push constant values
  if (!m_accumulate)
	  resetFrame();
  m_rtPushConstants.clearColor     = clearColor;

  m_rtPipeline->bind(cmdBuf);
  m_rtPipeline->bindDescriptorSets(cmdBuf, { m_rtDescSet, m_descSet });
  m_rtPipeline->pushConstant(cmdBuf, m_rtPushConstants);
  m_rtPipeline->trace(cmdBuf, { m_size.width, m_size.height, 1 });

  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// If the camera matrix has changed, resets the frame.
// otherwise, increments frame.
//
void HelloVulkan::updateFrame()
{
  static nvmath::mat4f refCamMatrix;
  static float         refFov{CameraManip.getFov()};

  const auto& m   = CameraManip.getMatrix();
  const auto  fov = CameraManip.getFov();

  if(memcmp(&refCamMatrix.a00, &m.a00, sizeof(nvmath::mat4f)) != 0 || refFov != fov)
  {
    resetFrame();
    refCamMatrix = m;
    refFov       = fov;
  }
  m_rtPushConstants.frame++;
}

void HelloVulkan::resetFrame()
{
  m_rtPushConstants.frame = -1;
}
