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

// ImGui - standalone example application for Glfw + Vulkan, using programmable
// pipeline If you are new to ImGui, see examples/README.txt and documentation
// at the top of imgui.cpp.

#include <array>
#include <iostream>
#include <vulkan/vulkan.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"

#include "hello_vulkan.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvpsystem.hpp"
#include "nvvk/appbase_vkpp.hpp"
#include "nvvk/commands_vk.hpp"

#include "RenderContext.h"
#include "FolderWatcher.h"

//////////////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)
//////////////////////////////////////////////////////////////////////////

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;

// Extra UI
void renderUI(HelloVulkan& helloVk)
{
  static int item = 1;
  if(ImGui::Combo("Up Vector", &item, "X\0Y\0Z\0\0"))
  {
	nvmath::vec3f pos, eye, up;
	CameraManip.getLookat(pos, eye, up);
	up = nvmath::vec3f(item == 0, item == 1, item == 2);
	CameraManip.setLookat(pos, eye, up);
  }
  helloVk.renderUI();
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
static int const SAMPLE_WIDTH  = 1280;
static int const SAMPLE_HEIGHT = 720;

//--------------------------------------------------------------------------------------------------
// Application Entry
//
int main(int argc, char** argv)
{
	//std::string fileName = "D:/repos/assets/glTF-Sample-Models/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf";
	std::string fileName = "D:/repos/assets/glTF-Sample-Models/2.0/SciFiHelmet/glTF/SciFiHelmet.gltf";
	//std::string fileName = "D:/repos/assets/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf";
	//std::string fileName = "D:/repos/assets/glTF-Sample-Models/2.0/TransmissionTest/glTF/TransmissionTest.gltf";
	//std::string fileName = "E:/repos/rev/samples/gltfViewer/bin/aventador/scene.gltf";
	if(argc > 1)
	{
		fileName = argv[1];
	}
	else
	{
		std::cout << "No scene filename provided\n";
		//return -1;
	}

	auto renderContext = RenderContext::create({SAMPLE_WIDTH, SAMPLE_HEIGHT}, "The other path tracer");
	if (!renderContext)
	{
		std::cout << "Failed to create render context\n";
		return -1;
	}

	// Setup camera
	CameraManip.setWindowSize(SAMPLE_WIDTH, SAMPLE_HEIGHT);
	CameraManip.setLookat(nvmath::vec3f(0, 0, 15), nvmath::vec3f(0, 0, 0), nvmath::vec3f(0, 1, 0));

	// setup some basic things for the sample, logging file for example
	NVPSystem system(argv[0], PROJECT_NAME);

	// Search path for shaders and other media
	defaultSearchPaths = {
		PROJECT_ABSDIRECTORY,        // shaders
		PROJECT_ABSDIRECTORY "../",  // media
		PROJECT_NAME,                // installed: shaders + media
		NVPSystem::exePath() + std::string(PROJECT_NAME),
	};

	// Create example
	HelloVulkan helloVk(*renderContext);

	helloVk.setup(
		renderContext->instance(),
		renderContext->device(),
		renderContext->physicalDevice(),
		renderContext->graphicsQueueIndex());

	helloVk.createSurface(renderContext->surface(), SAMPLE_WIDTH, SAMPLE_HEIGHT);
	helloVk.createDepthBuffer();
	helloVk.createRenderPass();
	helloVk.createFrameBuffers();

	// Setup Imgui
	helloVk.initGUI(0);  // Using sub-pass 0

	// Creation of the example
	//helloVk.loadScene(nvh::findFile("media/scenes/cornellBox.gltf", defaultSearchPaths));
	helloVk.loadScene(nvh::findFile(fileName, defaultSearchPaths));


	helloVk.createOffscreenRender();
	helloVk.createDescriptorSetLayout();
	helloVk.createGraphicsPipeline();
	helloVk.createUniformBuffer();
	helloVk.updateDescriptorSet();

	// #VKRay
	helloVk.initRayTracing();
	helloVk.createBottomLevelAS();
	helloVk.createTopLevelAS();
	helloVk.createRtDescriptorSet();
	helloVk.createRtPipeline();

	helloVk.createPostDescriptor();
	helloVk.createPostPipeline();
	helloVk.updatePostDescriptorSet();

	nvmath::vec4f clearColor   = nvmath::vec4f(1, 1, 1, 1.00f);
	bool          useRaytracer = true;

	auto window = renderContext->window();
	helloVk.setupGlfwCallbacks(window);
	ImGui_ImplGlfw_InitForVulkan(window, true);

	// Shader reload
	auto shadersFolder = std::string(PROJECT_ABSDIRECTORY) + "/shaders";
	auto shaderWatcher = FolderWatcher(std::filesystem::path(shadersFolder));
	shaderWatcher.listen([&helloVk](auto& changes) {
		for(auto& path : changes){
			if(path.string().find(".spv") != std::string::npos)
			{
				helloVk.invalidateShaders();
				helloVk.resetFrame();
				return;
			}
		}
	});

  // Main loop
  while(!glfwWindowShouldClose(window))
  {
    shaderWatcher.update();
	glfwPollEvents();
	if(helloVk.isMinimized())
	  continue;

	// Start the Dear ImGui frame
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	// Updating camera buffer
	helloVk.updateUniformBuffer();

	// Show UI window.
	if(1 == 1)
	{
	  ImGui::ColorEdit3("Clear color", reinterpret_cast<float*>(&clearColor));
	  ImGui::Checkbox("Ray Tracer mode", &useRaytracer);  // Switch between raster and ray tracing

	  renderUI(helloVk);
	  ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
				  1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	  ImGui::Render();
	}

	// Start rendering the scene
	helloVk.prepareFrame();

	// Start command buffer of this frame
	auto                     curFrame = helloVk.getCurFrame();
	const vk::CommandBuffer& cmdBuff  = helloVk.getCommandBuffers()[curFrame];

	cmdBuff.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

	// Clearing screen
	vk::ClearValue clearValues[2];
	clearValues[0].setColor(
		std::array<float, 4>({clearColor[0], clearColor[1], clearColor[2], clearColor[3]}));
	clearValues[1].setDepthStencil({1.0f, 0});

	// Offscreen render pass
	{
	  vk::RenderPassBeginInfo offscreenRenderPassBeginInfo;
	  offscreenRenderPassBeginInfo.setClearValueCount(2);
	  offscreenRenderPassBeginInfo.setPClearValues(clearValues);
	  offscreenRenderPassBeginInfo.setRenderPass(helloVk.m_offscreenRenderPass);
	  offscreenRenderPassBeginInfo.setFramebuffer(helloVk.m_offscreenFramebuffer);
	  offscreenRenderPassBeginInfo.setRenderArea({{}, helloVk.getSize()});

	  // Rendering Scene
	  if(useRaytracer)
	  {
		helloVk.raytrace(cmdBuff, clearColor);
	  }
	  else
	  {
		cmdBuff.beginRenderPass(offscreenRenderPassBeginInfo, vk::SubpassContents::eInline);
		helloVk.rasterize(cmdBuff);
		cmdBuff.endRenderPass();
	  }
	}

	// 2nd rendering pass: tone mapper, UI
	{
	  vk::RenderPassBeginInfo postRenderPassBeginInfo;
	  postRenderPassBeginInfo.setClearValueCount(2);
	  postRenderPassBeginInfo.setPClearValues(clearValues);
	  postRenderPassBeginInfo.setRenderPass(helloVk.getRenderPass());
	  postRenderPassBeginInfo.setFramebuffer(helloVk.getFramebuffers()[curFrame]);
	  postRenderPassBeginInfo.setRenderArea({{}, helloVk.getSize()});

	  cmdBuff.beginRenderPass(postRenderPassBeginInfo, vk::SubpassContents::eInline);
	  // Rendering tonemapper
	  helloVk.drawPost(cmdBuff);
	  // Rendering UI
	  ImGui::RenderDrawDataVK(cmdBuff, ImGui::GetDrawData());
	  cmdBuff.endRenderPass();
	}

	// Submit for display
	cmdBuff.end();
	helloVk.submitFrame();
  }

  // Cleanup
  helloVk.getDevice().waitIdle();
  helloVk.destroyResources();
  helloVk.destroy();

  return 0;
}
