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

#include <cassert>
#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>

#define NVVK_ALLOC_DEDICATED
#include "nvh/gltfscene.hpp"
#include "nvvk/allocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include <nvmath/nvmath_types.h>

#include "util.h"

class RenderScene
{
public:
  RenderScene(const vk::Device&         device,
              nvvk::AllocatorDedicated& alloc,
              nvvk::DebugUtil&,
              uint32_t gfxQueueNdx);
  ~RenderScene();

  void loadGltf(const std::string& fileName, nvmath::mat4f = nvmath::mat4f(1));

  void submitToGPU(const vk::CommandBuffer& cmdBuf);

  struct TextureTypeTag
  {
  };
  using TextureHandle = TagHandle<TextureTypeTag>;
  TextureHandle addTexture(const std::string& name, nvvk::Texture);

  const std::vector<nvvk::Texture>& textures() const { return m_textures; };

  const std::vector<vk::DescriptorImageInfo>& textureDescriptors() const
  {
    assert(m_textureDescriptors.size() == m_textures.size());
    return m_textureDescriptors;
  }

  static_assert(sizeof(nvh::GltfMaterial) % sizeof(nvmath::vec4f) == 0, "Materials need padding to a vec4");

  // --- CPU buffers ---
  std::vector<nvmath::mat4f> m_worldFromInstance;
  std::vector<nvh::GltfPrimMesh>     m_primitives;
  std::vector<uint32_t>      m_nodePrimitivesLUT;
  std::vector<nvh::GltfMaterial>       m_materials;

  // --- GPU buffers ---
  // Geometry
  nvvk::Buffer m_vtxPositionsBuffer;
  nvvk::Buffer m_normalsBuffer;
  nvvk::Buffer m_tangentsBuffer;
  nvvk::Buffer m_uvsBuffer;

  nvvk::Buffer m_indicesBuffer;

  // Materials
  nvvk::Buffer m_materialsBuffer;

  // Scene
  nvvk::Buffer m_primitivesBuffer;
  nvvk::Buffer m_instancePrimitivesBuffer;
  nvvk::Buffer m_worldFromInstanceBuffer;

  // Statistics
  size_t m_numVertices             = 0;
  size_t m_numTriangles            = 0;
  size_t m_maxVerticesPerPrimitive = 0;

private:
  void clearResources();
  // The command buffer may be used to allocate a dummy texture in case the scene doesn't contain any
  void updateTextureDescriptors();
  // We shouldn't bind empty arrays, so if there are no textures, we create a dummy one before submitting the scene
  void addDefaultTexture(const vk::CommandBuffer& cmdBuf);
  // Reverve space for n more textures
  void reserveTextures(size_t n);
  void createTextureImages(const vk::CommandBuffer& cmdBuf, tinygltf::Model& gltfModel);
  std::vector<vec4f> generateTangentSpace(const std::vector<vec3f>&    positions,
                                          const std::vector<vec3f>&    normals,
                                          const std::vector<vec2f>&    uvs,
                                          const std::vector<uint32_t>& indices,
                                          size_t                       nIndices,
                                          size_t                       nVertices,
                                          size_t                       indexOffset,
                                          size_t                       vertexOffset);

  std::vector<uint32_t> octEncodeVec3ToU32(const std::vector<nvmath::vec3>& normals);
  std::vector<uint32_t> packVec2ToU32(const std::vector<nvmath::vec2>& uvs);

  const vk::Device&         m_device;
  nvvk::AllocatorDedicated& m_alloc;
  nvvk::DebugUtil&          m_debug;
  uint32_t                  m_gfxQueueNdx{};

  std::vector<nvvk::Texture>           m_textures;
  std::vector<vk::DescriptorImageInfo> m_textureDescriptors;

  // Temporary storage buffers
  nvh::GltfScene             m_gltfScene;
  std::vector<nvmath::vec3f> m_vtxPositions;
  std::vector<uint32_t>      m_normals;
  std::vector<nvmath::vec4f> m_tangents;
  std::vector<uint32_t>      m_uvs;
  std::vector<uint16_t>      m_indices;
};
