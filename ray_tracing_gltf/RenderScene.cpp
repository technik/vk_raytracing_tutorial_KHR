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
// Copyright 2020 Carmelo J. Fernández-Agüera
#include "RenderScene.h"
#include <algorithm>

#include "nvvk/commands_vk.hpp"

RenderScene::RenderScene(const vk::Device&         device,
						 nvvk::AllocatorDedicated& alloc,
						 nvvk::DebugUtil&          debug,
						 uint32_t                  gfxQueueNdx)
	: m_device(device)
	, m_alloc(alloc)
	, m_debug(debug)
	, m_gfxQueueNdx(gfxQueueNdx)
{
}

RenderScene::~RenderScene()
{
  clearResources();
}

void RenderScene::loadGltf(const std::string& fileName, nvmath::mat4f rootTransform)
{
	if(fileName.find(".gltf") == fileName.npos)
		return;

	// Create the buffers on Device and copy vertices, indices and materials
	nvvk::CommandPool cmdBufGet(m_device, m_gfxQueueNdx);
	vk::CommandBuffer cmdBuf = cmdBufGet.createCommandBuffer();

	tinygltf::Model    tmodel;
	tinygltf::TinyGLTF tcontext;

	std::string warn, error;

	if(!tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, fileName))
		assert(!"Error while loading scene");

	m_gltfScene.importMaterials(tmodel);
	m_gltfScene.importDrawableNodes(tmodel,
									nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0);

	size_t primitiveOffset = m_primitives.size();
	for(const auto& node : m_gltfScene.m_nodes)
	{
		m_worldFromInstance.push_back(rootTransform * node.worldMatrix);
		m_nodePrimitivesLUT.push_back(node.primMesh + primitiveOffset);
	}

	// Textures
	size_t textureOffset = m_textures.size();
	createTextureImages(cmdBuf, tmodel);

	// Materials
	size_t materialOffset = m_materials.size();
	m_materials.reserve(m_gltfScene.m_materials.size());
	for(auto& gltfMaterial : m_gltfScene.m_materials)
	{
		m_materials.push_back(gltfMaterial);
		auto& material = m_materials.back();
		material.pbrBaseColorTexture = 
			gltfMaterial.pbrBaseColorTexture > -1 ?
				tmodel.textures[gltfMaterial.pbrBaseColorTexture].source + textureOffset :
				-1;
		material.pbrMetallicRoughnessTexture =
			gltfMaterial.pbrMetallicRoughnessTexture > -1 ?
				tmodel.textures[gltfMaterial.pbrMetallicRoughnessTexture].source + textureOffset :
				-1;
		material.emissiveTexture =
			gltfMaterial.emissiveTexture > -1 ?
				tmodel.textures[gltfMaterial.emissiveTexture].source + textureOffset :
				-1;
		material.normalTexture =
			gltfMaterial.normalTexture > -1 ?
				tmodel.textures[gltfMaterial.normalTexture].source + textureOffset :
				-1;

		m_materials.push_back(material);
	}

	size_t v0   = m_vtxPositions.size();
	size_t ndx0 = m_indices.size();

	// Copy vertex and index data
	m_vtxPositions.insert(m_vtxPositions.end(), m_gltfScene.m_positions.begin(),
						m_gltfScene.m_positions.end());
	std::vector<uint32_t> encodedNormals = octEncodeVec3ToU32(m_gltfScene.m_normals);
	m_normals.insert(m_normals.end(), encodedNormals.begin(), encodedNormals.end());
	std::vector<uint32_t> encodedUVs = packVec2ToU32(m_gltfScene.m_texcoords0);
	m_uvs.insert(m_uvs.end(), encodedUVs.begin(), encodedUVs.end());

	std::vector<uint16_t> shortIndices;
	shortIndices.reserve(m_gltfScene.m_indices.size());
	for(auto i : m_gltfScene.m_indices)
		m_indices.push_back(uint16_t(i));
	m_indices.insert(m_indices.end(), shortIndices.begin(), shortIndices.end());

	// Generate or extract tangent space
	std::vector<vec4f> tangents;
	if(m_gltfScene.m_tangents.size()
	!= m_gltfScene.m_positions.size())  // No tangents provided. Generate them
	{
		for(const auto& primitive : m_gltfScene.m_primMeshes)
		{
			tangents = generateTangentSpace(m_gltfScene.m_positions, m_gltfScene.m_normals,
											m_gltfScene.m_texcoords0, m_gltfScene.m_indices,
											primitive.indexCount, primitive.vertexCount,
											primitive.firstIndex, primitive.vertexOffset);
			m_tangents.insert(m_tangents.end(), tangents.begin(), tangents.end());
		}
	}
	else
		m_tangents.insert(m_tangents.end(), m_gltfScene.m_tangents.begin(),
					  m_gltfScene.m_tangents.end());

	// Store primitive look up tables
	m_primitives.reserve(m_gltfScene.m_primMeshes.size());
	for(const auto& gltfPrimitive : m_gltfScene.m_primMeshes)
	{
		m_numVertices += gltfPrimitive.vertexCount;
		m_numTriangles += gltfPrimitive.indexCount;
		m_maxVerticesPerPrimitive = std::max<int32_t>(m_maxVerticesPerPrimitive, gltfPrimitive.vertexCount);

		m_primitives.push_back(gltfPrimitive);
		auto& primitive = m_primitives.back();

		primitive.firstIndex    = primitive.firstIndex + ndx0;
		primitive.vertexOffset  = primitive.vertexOffset + v0;
		primitive.indexCount    = primitive.indexCount;
		primitive.vertexCount   = primitive.vertexCount;
		primitive.materialIndex = primitive.materialIndex + materialOffset;
	}

	cmdBufGet.submitAndWait(cmdBuf);

	m_gltfScene.destroy();  // Release buffers
}

//--------------------------------------------------------------------------------------------------
// Creating all textures and samplers (gltf version)
//
void RenderScene::createTextureImages(const vk::CommandBuffer& cmdBuf, tinygltf::Model& gltfModel)
{
	using vkIU = vk::ImageUsageFlagBits;

	vk::SamplerCreateInfo samplerCreateInfo{
		{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
	samplerCreateInfo.setMaxLod(FLT_MAX);
	//vk::Format format = vk::Format::eR8G8B8A8Srgb;
	vk::Format format = vk::Format::eR8G8B8A8Unorm;

	reserveTextures(gltfModel.images.size());
	for(size_t i = 0; i < gltfModel.images.size(); i++)
	{
	auto&        gltfimage  = gltfModel.images[i];
	void*        buffer     = &gltfimage.image[0];
	VkDeviceSize bufferSize = gltfimage.image.size();
	auto         imgSize    = vk::Extent2D(gltfimage.width, gltfimage.height);

	if(bufferSize == 0 || gltfimage.width == -1 || gltfimage.height == -1)
	{
		assert(false);
	}

	vk::ImageCreateInfo imageCreateInfo =
		nvvk::makeImage2DCreateInfo(imgSize, format, vkIU::eSampled, true);

	nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, buffer, imageCreateInfo);
	nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
	vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);

	// Include image index in debug texture name in case the gltf name was empty
	auto debugTextureName = std::string("Txt" + std::to_string(i) + gltfModel.images[i].name);
	addTexture(debugTextureName, m_alloc.createTexture(image, ivInfo, samplerCreateInfo));
	}
}

void RenderScene::submitToGPU(const vk::CommandBuffer& cmdBuf)
{
	if(m_textures.empty())
	addDefaultTexture(cmdBuf);

	updateTextureDescriptors();

	// Geometry
	using vkBU = vk::BufferUsageFlagBits;
	using vkMP = vk::MemoryPropertyFlagBits;

	m_vtxPositionsBuffer =
		m_alloc.createBuffer(cmdBuf, m_vtxPositions,
							vkBU::eVertexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress);
	m_normalsBuffer =
		m_alloc.createBuffer(cmdBuf, m_normals, vkBU::eVertexBuffer | vkBU::eStorageBuffer);
	m_tangentsBuffer =
		m_alloc.createBuffer(cmdBuf, m_tangents, vkBU::eVertexBuffer | vkBU::eStorageBuffer);
	m_uvsBuffer = m_alloc.createBuffer(cmdBuf, m_uvs, vkBU::eVertexBuffer | vkBU::eStorageBuffer);

	m_indicesBuffer =
		m_alloc.createBuffer(cmdBuf, m_indices,
							vkBU::eIndexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress);

	// Materials
	m_materialsBuffer =
		m_alloc.createBuffer(cmdBuf, m_materials, vkBU::eStorageBuffer,
							vk::MemoryPropertyFlags(vkMP::eHostVisible | vkMP::eHostCoherent));

	// Scene representation
	m_worldFromInstanceBuffer =
		m_alloc.createBuffer(cmdBuf, m_worldFromInstance, vkBU::eStorageBuffer);
	m_primitivesBuffer = m_alloc.createBuffer(cmdBuf, m_primitives, vkBU::eStorageBuffer);
	m_instancePrimitivesBuffer =
		m_alloc.createBuffer(cmdBuf, m_nodePrimitivesLUT, vkBU::eStorageBuffer);

	// Set debug names
	m_debug.setObjectName(m_vtxPositionsBuffer.buffer, "vertex pos");
	m_debug.setObjectName(m_normalsBuffer.buffer, "normals");
	m_debug.setObjectName(m_tangentsBuffer.buffer, "tangents");
	m_debug.setObjectName(m_uvsBuffer.buffer, "uvs");
	m_debug.setObjectName(m_indicesBuffer.buffer, "indices");
	m_debug.setObjectName(m_worldFromInstanceBuffer.buffer, "worldFromInstance");
	m_debug.setObjectName(m_materialsBuffer.buffer, "materials");
	m_debug.setObjectName(m_primitivesBuffer.buffer, "primitives");
	m_debug.setObjectName(m_instancePrimitivesBuffer.buffer, "instancePrimitives");

	// Clear temporary buffers
	m_vtxPositions.clear();
	m_normals.clear();
	m_tangents.clear();
	m_uvs.clear();
	m_indices.clear();
}

void RenderScene::clearResources()
{
	// Textures
	for(auto& texture : m_textures)
		m_alloc.destroy(texture);
	m_textures.clear();

	// Geometry
	m_alloc.destroy(m_vtxPositionsBuffer);
	m_alloc.destroy(m_normalsBuffer);
	m_alloc.destroy(m_tangentsBuffer);
	m_alloc.destroy(m_uvsBuffer);
	m_alloc.destroy(m_indicesBuffer);

	m_alloc.destroy(m_materialsBuffer);

	// Scene description
	m_alloc.destroy(m_primitivesBuffer);
	m_alloc.destroy(m_worldFromInstanceBuffer);
	m_alloc.destroy(m_instancePrimitivesBuffer);
}

void RenderScene::updateTextureDescriptors()
{
	assert(!m_textures.empty());  // Make sure we won't try to bind an empty array
	assert(m_textureDescriptors.empty());
	m_textureDescriptors.reserve(m_textures.size());
	for(size_t i = 0; i < m_textures.size(); ++i)
	{
		m_textureDescriptors.push_back(m_textures[i].descriptor);
	}
}

auto RenderScene::addTexture(const std::string& name, nvvk::Texture nvvkTexture) -> TextureHandle
{
	m_textures.push_back(nvvkTexture);
	return TextureHandle(m_textures.size() - 1);
	m_debug.setObjectName(nvvkTexture.image, name);
}

void RenderScene::addDefaultTexture(const vk::CommandBuffer& cmdBuf)
{
	// Make dummy image(1,1), needed as we cannot have an empty array
	std::array<uint8_t, 4> white = {255, 255, 255, 255};
	addTexture("white-dummy",
				m_alloc.createTexture(cmdBuf, 4, white.data(),
									nvvk::makeImage2DCreateInfo(vk::Extent2D{1, 1}), {}));
}

void RenderScene::reserveTextures(size_t n)
{
  m_textures.reserve(m_textures.size() + n);
}

//----------------------------------------------------------------------------------------------
std::vector<vec4f> RenderScene::generateTangentSpace(
	const std::vector<vec3f>&    positions,
	const std::vector<vec3f>&    normals,
	const std::vector<vec2f>&    uvs,
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
	for(int i = indexOffset; i < indexEnd; i += 3)  // Iterate over all triangles
	{
		auto i0 = indices[i + 0];
		auto i1 = indices[i + 1];
		auto i2 = indices[i + 2];

		vec2f localUvs[3] = {uvs[i0 + vertexOffset], uvs[i1 + vertexOffset], uvs[i2 + vertexOffset]};
		vec3f localPos[3] = {positions[i0 + vertexOffset], positions[i1 + vertexOffset],
								positions[i2 + vertexOffset]};

		vec2f deltaUV1 = localUvs[1] - localUvs[0];
		vec2f deltaUV2 = localUvs[2] - localUvs[0];

		vec3f deltaPos1 = localPos[1] - localPos[0];
		vec3f deltaPos2 = localPos[2] - localPos[0];

		auto determinant = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;

		// Unnormalized tangent
		vec3f triangleTangent = (deltaPos1 * deltaUV2.y - deltaUV1.y * deltaPos2) * (1 / determinant);

		tangentVectors[i0] +=
			vec4f(triangleTangent.x, triangleTangent.y, triangleTangent.z, determinant);
		tangentVectors[i1] +=
			vec4f(triangleTangent.x, triangleTangent.y, triangleTangent.z, determinant);
		tangentVectors[i2] +=
			vec4f(triangleTangent.x, triangleTangent.y, triangleTangent.z, determinant);
	}

	// Orthonormalize per vertex
	for(int i = 0; i < tangentVectors.size(); ++i)
	{
		auto& tangent  = tangentVectors[i];
		vec3f tangent3 = {tangent.x, tangent.y, tangent.z};
		auto& normal   = normals[i];

		tangent3 = tangent3 - (dot(tangent3, normal) * normal);  // Orthogonal tangent
		tangent3 = normalize(tangent3);                          // Orthonormal tangent
		tangent  = {tangent3.x, tangent3.y, tangent3.z, signbit(-tangent.w) ? -1.f : 1.f};
	}

	return tangentVectors;
}

vec2 signNotZero(vec2 v)
{
	return vec2(v.x < 0 ? -1.0 : 1.0, v.y < 0 ? -1.0 : 1.0);
}

// Compress a vec3 unit vector using octahedron encoding, and store it in two
// snorm16 packed into a single uint32
uint32_t octEncodeUnitVector(const nvmath::vec3& v)
{
	// Project the sphere onto the octahedron
	// Ignore the z component, as it can be trivially reconstructed as z=1-x-y
	vec2 p = vec2(v) * (1.f / (abs(v.x) + abs(v.y) + abs(v.z)));
	if(v.z < 0)
		p = (vec2(1.f) - vec2(abs(p.y), abs(p.x))) * signNotZero(p);
	vec2 unpacked = (p * 0.5f + 0.5f) * float((1 << 16) - 2);
	return uint16_t(unpacked.x) | (uint32_t(unpacked.y) << 16);
}

std::vector<uint32_t> RenderScene::octEncodeVec3ToU32(const std::vector<nvmath::vec3>& normals)
{
	std::vector<uint32_t> encoded;
	encoded.reserve(normals.size());
	for(auto& v : normals)
	{
		encoded.push_back(octEncodeUnitVector(v));
	}
	return encoded;
}

std::vector<uint32_t> RenderScene::packVec2ToU32(const std::vector<nvmath::vec2>& uvs)
{
	std::vector<uint32_t> encoded;
	encoded.reserve(uvs.size());
	for(auto& v : uvs)
	{
		vec2     scaled = v * float((1 << 16) - 1) + 0.5f;  // Scale and round
		uint32_t packed = uint16_t(scaled.x) | (uint32_t(scaled.y) << 16);
		encoded.push_back(packed);
	}
	return encoded;
}