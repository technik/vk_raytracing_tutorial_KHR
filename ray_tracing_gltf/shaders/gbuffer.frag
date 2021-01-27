#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

#include "binding.glsl"
#include "gltf.glsl"
#include "pbr.glsl"


layout(push_constant) uniform shaderInformation
{
  vec3  lightPosition;
  uint  instanceId;
  float lightIntensity;
  int   lightType;
  int   matetrialId;
}
pushC;

// clang-format off
// Incoming 
//layout(location = 0) flat in int matIndex;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec4 fragTan;
layout(location = 4) in vec3 viewDir;
layout(location = 5) in vec3 worldPos;
// Outgoing
layout(location = 0) out vec4 outBaseColor;
layout(location = 1) out vec4 outNormals;
layout(location = 2) out vec4 outPBR;
layout(location = 3) out vec4 outEmissive;
// Buffers
layout(set = 0, binding = B_MATERIALS) buffer _GltfMaterial { GltfShadeMaterial materials[]; };
layout(set = 0, binding = B_TEXTURES) uniform sampler2D[] textureSamplers;

// clang-format on


void main()
{
  vec3 N = normalize(fragNormal);
  vec4 msTangent = fragTan;
  msTangent.xyz = normalize(msTangent.xyz);

  bool albedo85 = false;
  FragmentInfo surfaceProperties = sampleMaterial(
    pushC.matetrialId,
    fragTexCoord,
    N,
    msTangent,
    albedo85);

  // Material of the object
  GltfShadeMaterial mat = materials[nonuniformEXT(pushC.matetrialId)];


  // Vector toward light
  vec3  L;
  float lightIntensity = pushC.lightIntensity;
  if(pushC.lightType == 0)
  {
    vec3  lDir     = pushC.lightPosition - worldPos;
    float d        = length(lDir);
    lightIntensity = pushC.lightIntensity / (d * d);
    L              = normalize(lDir);
  }
  else
  {
    L = normalize(pushC.lightPosition - vec3(0));
  }


  // Diffuse
  vec3 diffuse = computeDiffuse(mat, L, N);
  if(mat.pbrBaseColorTexture > -1)
  {
    uint txtId      = mat.pbrBaseColorTexture;
    vec3 diffuseTxt = texture(textureSamplers[nonuniformEXT(txtId)], fragTexCoord).xyz;
    diffuse *= diffuseTxt;
  }

  // Specular
  vec3 specular = computeSpecular(mat, viewDir, L, N);

  // Result
  outEmissive = vec4(lightIntensity * (diffuse + specular), 1);
  outNormals = vec4(N, 1.0);
  outPBR = vec4(mat.metallic, mat.roughness, 0, 1);
  outBaseColor = vec4(diffuse, 1);
}
