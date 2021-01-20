#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#include "binding.glsl"
#include "gltf.glsl"
#include "raycommon.glsl"
#include "sampling.glsl"


hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(location = 1) rayPayloadEXT bool isShadowed;

layout(set = 0, binding = 0 ) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 2) readonly buffer _InstanceInfo {PrimMeshInfo primInfo[];};

layout(set = 1, binding = B_VERTICES) readonly buffer _VertexBuf {float vertices[];};
layout(set = 1, binding = B_INDICES) readonly buffer _Indices {uint indices[];};
layout(set = 1, binding = B_NORMALS) readonly buffer _NormalBuf {float normals[];};
layout(set = 1, binding = B_TEXCOORDS) readonly buffer _TexCoordBuf {float texcoord0[];};
layout(set = 1, binding = B_MATERIALS) readonly buffer _MaterialBuffer {GltfShadeMaterial materials[];};
layout(set = 1, binding = B_TEXTURES) uniform sampler2D texturesMap[]; // all textures


layout(push_constant) uniform Constants
{
    vec4  clearColor;
    int   frame;
    float lensRadius;
    float focalDistance;
    int maxBounces;
    int firstBounce;
    int numLightInstances;
    uint renderFlags;
}
pushC;

bool renderFlag(uint flag)
{
  return (pushC.renderFlags & flag) != 0;
}

// Return the vertex position
vec3 getVertex(uint index)
{
  vec3 vp;
  vp.x = vertices[3 * index + 0];
  vp.y = vertices[3 * index + 1];
  vp.z = vertices[3 * index + 2];
  return vp;
}

vec3 getNormal(uint index)
{
  vec3 vp;
  vp.x = normals[3 * index + 0];
  vp.y = normals[3 * index + 1];
  vp.z = normals[3 * index + 2];
  return vp;
}

vec2 getTexCoord(uint index)
{
  vec2 vp;
  vp.x = texcoord0[2 * index + 0];
  vp.y = texcoord0[2 * index + 1];
  return vp;
}


void main()
{
  // Retrieve the Primitive mesh buffer information
  PrimMeshInfo pinfo = primInfo[gl_InstanceCustomIndexEXT];

  // Getting the 'first index' for this mesh (offset of the mesh + offset of the triangle)
  uint indexOffset  = pinfo.indexOffset + (3 * gl_PrimitiveID);
  uint vertexOffset = pinfo.vertexOffset;           // Vertex offset as defined in glTF
  uint matIndex     = max(0, pinfo.materialIndex);  // material of primitive mesh

  // Getting the 3 indices of the triangle (local)
  ivec3 triangleIndex = ivec3(indices[nonuniformEXT(indexOffset + 0)],  //
                              indices[nonuniformEXT(indexOffset + 1)],  //
                              indices[nonuniformEXT(indexOffset + 2)]);
  triangleIndex += ivec3(vertexOffset);  // (global)

  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Vertex of the triangle
  const vec3 pos0           = getVertex(triangleIndex.x);
  const vec3 pos1           = getVertex(triangleIndex.y);
  const vec3 pos2           = getVertex(triangleIndex.z);
  const vec3 position       = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  const vec3 world_position = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

  // Normal
  const vec3 nrm0 = getNormal(triangleIndex.x);
  const vec3 nrm1 = getNormal(triangleIndex.y);
  const vec3 nrm2 = getNormal(triangleIndex.z);
  vec3 normal = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  const vec3 world_normal = normalize(vec3(normal * gl_WorldToObjectEXT));
  const vec3 geom_normal  = normalize(cross(pos1 - pos0, pos2 - pos0));

  // TexCoord
  const vec2 uv0       = getTexCoord(triangleIndex.x);
  const vec2 uv1       = getTexCoord(triangleIndex.y);
  const vec2 uv2       = getTexCoord(triangleIndex.z);
  vec2 texcoord0 = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

  prd.worldPos.xyz = world_position;
  prd.worldPos.w = gl_HitTEXT;
  prd.worldNormal = world_normal;

  // Material of the object
  if(matIndex >= 0)
  {
      GltfShadeMaterial mat = materials[nonuniformEXT(matIndex)];
      // Emissive color
      prd.emittance = mat.emissiveFactor;
      if(mat.emissiveTexture > -1)
      {
          uint txtId = mat.emissiveTexture;
          prd.emittance *= texture(texturesMap[nonuniformEXT(txtId)], texcoord0).xyz;
      }
      // baseColor
      prd.baseColor = mat.pbrBaseColorFactor.xyz;
      if(mat.pbrBaseColorTexture > -1)
      {
          uint txtId = mat.pbrBaseColorTexture;
          prd.baseColor *= texture(texturesMap[nonuniformEXT(txtId)], texcoord0).xyz;
      }

      // Metallic & Roughness
      /*prd.metallic = mat.pbrMetallicFactor;
      prd.roughness = mat.pbrRoughnessFactor;
      if(mat.pbrMetallicRoughnessTexture > -1)
      {
          uint txtId = mat.pbrMetallicRoughnessTexture;
          vec3 metallicRoughness = texture(texturesMap[nonuniformEXT(txtId)], texcoord0).xyz;
          prd.metallic *= metallicRoughness.b;
          prd.roughness *= metallicRoughness.g;
      }*/
  }
  else
  {
      prd.baseColor = vec3(1);
      prd.emittance = vec3(0);
      prd.metallic = 0.0;
      prd.roughness = 1.0;
  }
  if(renderFlag(FLAG_ALBEDO_85))
  {
    prd.baseColor = vec3(0.85);
  }
}
