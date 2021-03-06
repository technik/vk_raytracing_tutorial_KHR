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
#include "nvh/gltfscene.hpp"
#include "nvh/nvprint.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"

#include "RenderContext.h"
#include "RenderScene.h"
#include "shaders/binding.glsl"

// Holding the camera matrices
struct CameraMatrices
{
  nvmath::mat4f view;
  nvmath::mat4f proj;
  nvmath::mat4f viewInverse;
  // #VKRay
  nvmath::mat4f projInverse;
};

HelloVulkan::HelloVulkan(RenderContext& ctxt)
    : m_alloc(ctxt.alloc())
{
}

//--------------------------------------------------------------------------------------------------
void HelloVulkan::renderUI()
{
	bool mustClean = false;
	ImGui::Checkbox("Accumulate", &m_accumulate);
	mustClean |= ImGui::SliderFloat3("Light Position", &m_pushConstant.lightPosition.x, -20.f, 20.f);
	mustClean |= ImGui::SliderFloat("Sky Intensity", &m_rtPushConstants.skyIntensity, 0.f, 100.f);
	mustClean |= ImGui::SliderFloat("Sun Intensity", &m_rtPushConstants.sunIntensity, 0.f, 100.f);
	if (ImGui::CollapsingHeader("Reference path tracer"))
	{
		mustClean |= ImGui::InputInt("Max bounces", &m_rtPushConstants.maxBounces, 1);
		mustClean |= ImGui::InputInt("First bounce", &m_rtPushConstants.firstBounce, 1);
		m_rtPushConstants.maxBounces = std::min(20, std::max(0, m_rtPushConstants.maxBounces));
		m_rtPushConstants.firstBounce = std::min(20, std::max(0, m_rtPushConstants.firstBounce));
		// Render flags
		bool overrideAlbedo = m_rtPushConstants.renderFlags & (1 << 1);
		mustClean |= ImGui::Checkbox("Albedo 0.85", &overrideAlbedo);
		bool greyFurnace = m_rtPushConstants.renderFlags & (1 << 3);
		mustClean |= ImGui::Checkbox("Furnace test", &greyFurnace);
		bool diffuseOnly = m_rtPushConstants.renderFlags & (1 << 4);
		mustClean |= ImGui::Checkbox("Ignore specular", &diffuseOnly);
		bool specularOnly = m_rtPushConstants.renderFlags & (1 << 5);
		mustClean |= ImGui::Checkbox("Ignore diffuse", &specularOnly);
		bool importanceSampling = m_rtPushConstants.renderFlags & (1 << 6);
		mustClean |= ImGui::Checkbox("Importance sampling", &importanceSampling);
		bool useDOF = m_rtPushConstants.renderFlags & (1 << 7);
		mustClean |= ImGui::Checkbox("Depth of field", &useDOF);
		m_rtPushConstants.renderFlags =
			(overrideAlbedo ? (1 << 1) : 0) |
			(greyFurnace ? (1 << 3) : 0) |
			(diffuseOnly ? (1 << 4) : 0) |
			(specularOnly ? (1 << 5) : 0) |
			(importanceSampling ? (1 << 6) : 0) |
			(useDOF ? (1 << 7) : 0);
		if(useDOF)
		{
		  float expFocalDistance = log10f(m_rtPushConstants.focalDistance);
		  mustClean |= ImGui::SliderFloat("Focal distance exp", &expFocalDistance, -3.f, 3.f);
		  mustClean |= ImGui::SliderFloat("Lens radius", &m_rtPushConstants.lensRadius, 0.f, 0.5f);
		  m_rtPushConstants.focalDistance = powf(10.f, expFocalDistance);
		}
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
void HelloVulkan::updateUniformBuffer()
{
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);

  CameraMatrices ubo = {};
  ubo.view           = CameraManip.getMatrix();
  ubo.proj           = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
  // ubo.proj[1][1] *= -1;  // Inverting Y for Vulkan
  ubo.viewInverse = nvmath::invert(ubo.view);
  // #VKRay
  ubo.projInverse = nvmath::invert(ubo.proj);

  void* data = m_device.mapMemory(m_cameraMat.allocation, 0, sizeof(ubo));
  memcpy(data, &ubo, sizeof(ubo));
  m_device.unmapMemory(m_cameraMat.allocation);
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
  bind.addBinding(
	  vkDS(B_VERTICES, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));
  bind.addBinding(
	  vkDS(B_INDICES, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));
  bind.addBinding(vkDS(B_NORMALS, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR));
  bind.addBinding(vkDS(B_TANGENTS, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR));
  bind.addBinding(vkDS(B_TEXCOORDS, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));
  bind.addBinding(vkDS(B_MATERIALS, vkDT::eStorageBuffer, 1,
					   vkSS::eFragment | vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));
  bind.addBinding(vkDS(B_MATRICES, vkDT::eStorageBuffer, 1,
					   vkSS::eVertex | vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));
  auto nbTextures = static_cast<uint32_t>(m_textures.size());
  bind.addBinding(vkDS(B_TEXTURES, vkDT::eCombinedImageSampler, nbTextures,
					   vkSS::eFragment | vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));


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
  vk::DescriptorBufferInfo matrixDesc{m_matrixBuffer.buffer, 0, VK_WHOLE_SIZE};

  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_CAMERA, &dbiUnif));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_VERTICES, &vertexDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_INDICES, &indexDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_NORMALS, &normalDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_TANGENTS, &tangentDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_TEXCOORDS, &uvDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_MATERIALS, &materialDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_MATRICES, &matrixDesc));

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
  gpb.addShader(nvh::loadFile("shaders/vert_shader.vert.spv", true, paths), vkSS::eVertex);
  gpb.addShader(nvh::loadFile("shaders/frag_shader.frag.spv", true, paths), vkSS::eFragment);
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

//--------------------------------------------------------------------------------------------------
// Loading the OBJ file and setting up all buffers
//
void HelloVulkan::loadScene(const std::string& filename)
{
  using vkBU = vk::BufferUsageFlagBits;
  tinygltf::Model    tmodel;
  tinygltf::TinyGLTF tcontext;
  std::string        warn, error;

  if(!tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, filename))
  {
	assert(!"Error while loading scene");
  }
  LOGW(warn.c_str());
  LOGE(error.c_str());


  m_gltfScene.importMaterials(tmodel);
  m_gltfScene.importDrawableNodes(tmodel,
								  nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0);

  // Create the buffers on Device and copy vertices, indices and materials
  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  vk::CommandBuffer cmdBuf = cmdBufGet.createCommandBuffer();

  m_vertexBuffer =
	  m_alloc.createBuffer(cmdBuf, m_gltfScene.m_positions,
						   vkBU::eVertexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress);
  m_indexBuffer =
	  m_alloc.createBuffer(cmdBuf, m_gltfScene.m_indices,
						   vkBU::eIndexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress);
  m_normalBuffer   = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_normals,
										vkBU::eVertexBuffer | vkBU::eStorageBuffer);
  m_uvBuffer       = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_texcoords0,
									vkBU::eVertexBuffer | vkBU::eStorageBuffer);
  m_materialBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_materials, vkBU::eStorageBuffer);

  // Generate or load tangent space
  std::vector<vec4f> tangents;
  if(m_gltfScene.m_tangents.size()
     != m_gltfScene.m_positions.size())  // No tangents provided. Generate them
  {
    m_gltfScene.m_tangents.clear();
    for(const auto& primitive : m_gltfScene.m_primMeshes)
    {
      tangents = RenderScene::generateTangentSpace(m_gltfScene.m_positions, m_gltfScene.m_normals,
                                      m_gltfScene.m_texcoords0, m_gltfScene.m_indices,
                                      primitive.indexCount, primitive.vertexCount,
                                      primitive.firstIndex, primitive.vertexOffset);
      m_gltfScene.m_tangents.insert(m_gltfScene.m_tangents.end(), tangents.begin(), tangents.end());
    }
  }

  tangents = m_gltfScene.m_tangents;
  m_tangentBuffer = m_alloc.createBuffer(cmdBuf, tangents,
                                        vkBU::eVertexBuffer | vkBU::eStorageBuffer);

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
	primLookup.push_back({primMesh.firstIndex, primMesh.vertexOffset, primMesh.materialIndex});
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
}


//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer()
{
  using vkBU = vk::BufferUsageFlagBits;
  using vkMP = vk::MemoryPropertyFlagBits;

  m_cameraMat = m_alloc.createBuffer(sizeof(CameraMatrices), vkBU::eUniformBuffer,
									 vkMP::eHostVisible | vkMP::eHostCoherent);
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
  m_referencePTPipeline = nullptr;
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
  vk::PushConstantRange pushConstantRanges = {vk::ShaderStageFlagBits::eFragment, 0, sizeof(float)};

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
  pipelineGenerator.addShader(nvh::loadFile("shaders/passthrough.vert.spv", true, paths),
							  vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("shaders/post.frag.spv", true, paths),
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

  auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
  cmdBuf.pushConstants<float>(m_postPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0,
							  aspectRatio);
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
  auto properties = m_physicalDevice.getProperties2<vk::PhysicalDeviceProperties2,
													vk::PhysicalDeviceRayTracingPropertiesKHR>();
  m_rtProperties  = properties.get<vk::PhysicalDeviceRayTracingPropertiesKHR>();
  m_rtBuilder.setup(m_device, &m_alloc, m_graphicsQueueIndex);
}

//--------------------------------------------------------------------------------------------------
// Converting a GLTF primitive in the Raytracing Geometry used for the BLAS
//
nvvk::RaytracingBuilderKHR::Blas HelloVulkan::primitiveToGeometry(const nvh::GltfPrimMesh& prim)
{
  // Setting up the creation info of acceleration structure
  vk::AccelerationStructureCreateGeometryTypeInfoKHR asCreate;
  asCreate.setGeometryType(vk::GeometryTypeKHR::eTriangles);
  asCreate.setIndexType(vk::IndexType::eUint32);
  asCreate.setVertexFormat(vk::Format::eR32G32B32Sfloat);
  asCreate.setMaxPrimitiveCount(prim.indexCount / 3);  // Nb triangles
  asCreate.setMaxVertexCount(prim.vertexCount);
  asCreate.setAllowsTransforms(VK_FALSE);  // No adding transformation matrices

  // Building part
  vk::DeviceAddress vertexAddress = m_device.getBufferAddress({m_vertexBuffer.buffer});
  vk::DeviceAddress indexAddress  = m_device.getBufferAddress({m_indexBuffer.buffer});

  vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
  triangles.setVertexFormat(asCreate.vertexFormat);
  triangles.setVertexData(vertexAddress);
  triangles.setVertexStride(sizeof(nvmath::vec3f));
  triangles.setIndexType(asCreate.indexType);
  triangles.setIndexData(indexAddress);
  triangles.setTransformData({});

  // Setting up the build info of the acceleration
  vk::AccelerationStructureGeometryKHR asGeom;
  asGeom.setGeometryType(asCreate.geometryType);
  asGeom.setFlags(vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation);  // For AnyHit
  asGeom.geometry.setTriangles(triangles);


  vk::AccelerationStructureBuildOffsetInfoKHR offset;
  offset.setFirstVertex(prim.vertexOffset);
  offset.setPrimitiveCount(prim.indexCount / 3);
  offset.setPrimitiveOffset(prim.firstIndex * sizeof(uint32_t));
  offset.setTransformOffset(0);

  nvvk::RaytracingBuilderKHR::Blas blas;
  blas.asGeometry.emplace_back(asGeom);
  blas.asCreateGeometryInfo.emplace_back(asCreate);
  blas.asBuildOffsetInfo.emplace_back(offset);
  return blas;
}

//--------------------------------------------------------------------------------------------------
//
//
void HelloVulkan::createBottomLevelAS()
{
  // BLAS - Storing each primitive in a geometry
  std::vector<nvvk::RaytracingBuilderKHR::Blas> allBlas;
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
	rayInst.transform  = node.worldMatrix;
	rayInst.instanceId = node.primMesh;  // gl_InstanceCustomIndexEXT: to find which primitive
	rayInst.blasId     = node.primMesh;
	rayInst.flags      = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
	rayInst.hitGroupId = 0;  // We will use the same hit group for all objects
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
	  2, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));  // Primitive info

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
	std::vector<std::string> rayGenShaders = {"shaders/pathtrace.rgen.spv"};
	std::vector<std::string> missShaders = {"shaders/pathtrace.rmiss.spv"};
	std::vector<std::string> chitShaders = {"shaders/pathtrace.rchit.spv"};
	std::vector<std::string> anyHitShaders = {"shaders/pathtrace.rahit.spv"};

	RaytracingPipeline::PipelineLayoutInfo pipelineLayout;
	pipelineLayout.descSetLayouts = {m_rtDescSetLayout, m_descSetLayout};
	pipelineLayout.pushConstantRangeSize = sizeof(RtPushConstant);

	m_referencePTPipeline = std::make_unique<RaytracingPipeline>(
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
  m_rtPushConstants.clearColor     = clearColor;
  m_rtPushConstants.lightPosition  = m_pushConstant.lightPosition;
  m_rtPushConstants.lightPosition.normalize();

  m_referencePTPipeline->bind(cmdBuf);
  m_referencePTPipeline->bindDescriptorSets(cmdBuf, {m_rtDescSet, m_descSet});
  m_referencePTPipeline->pushConstant(cmdBuf, m_rtPushConstants);
  m_referencePTPipeline->trace(cmdBuf, { m_size.width, m_size.height, 1 });

  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// If the camera matrix has changed, resets the frame.
// otherwise, increments frame.
//
void HelloVulkan::updateFrame()
{
  static nvmath::mat4f refCamMatrix;

  auto& m = CameraManip.getMatrix();
  if(memcmp(&refCamMatrix.a00, &m.a00, sizeof(nvmath::mat4f)) != 0)
  {
	resetFrame();
	refCamMatrix = m;
  }
  m_rtPushConstants.frame++;
}

void HelloVulkan::resetFrame()
{
  m_rtPushConstants.frame = -1;
}
