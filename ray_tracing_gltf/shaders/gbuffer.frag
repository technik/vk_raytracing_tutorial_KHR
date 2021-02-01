#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

#include "binding.glsl"
#include "gltf.glsl"


layout(push_constant) uniform shaderInformation
{
  vec3  lightPosition;
  uint  instanceId;
  float lightIntensity;
  int   lightType;
  int   matetrialId;
}
pushC;

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
layout(set = 0, binding = B_TEXTURES) uniform sampler2D[] texturesMap;

struct FragmentInfo
{
  vec3 emittance;
  vec3 baseColor;
  float metallic;
  float roughness;
  vec3 msNormal;
};

FragmentInfo sampleMaterial(int matIndex, in vec2 texcoord0, in vec3 msNormal, in vec4 msTangent, bool albedo85)
{
  FragmentInfo result;
  result.msNormal = msNormal;
  if(matIndex >= 0)
    {
        GltfShadeMaterial mat = materials[nonuniformEXT(matIndex)];
        texcoord0 = texcoord0*mat.scale+mat.offset;
        // Emissive color
        result.emittance = mat.emissiveFactor;
        if(mat.emissiveTexture > -1)
        {
            uint txtId = mat.emissiveTexture;
            result.emittance *= texture(texturesMap[nonuniformEXT(txtId)], texcoord0).xyz * 10;
        }
        // baseColor
        result.baseColor    = mat.pbrBaseColorFactor.xyz;
        if(mat.pbrBaseColorTexture > -1)
        {
            uint txtId = mat.pbrBaseColorTexture;
            result.baseColor *= texture(texturesMap[nonuniformEXT(txtId)], texcoord0).xyz;
        }

        /*
        if(mat.normalTexture >= 0)
        {
            vec3 msBitangent = cross(msNormal, msTangent.xyz) * msTangent.w;

            mat3 modelFromTangent = mat3(msTangent, msBitangent, msNormal);

            uint txtId = mat.normalTexture;
            vec3 tsNormal = texture(texturesMap[nonuniformEXT(txtId)], texcoord0).xyz;
            tsNormal = pow(tsNormal, vec3(1/2.2)) + vec3(0,0,1e-2);
            tsNormal = normalize(tsNormal * 255.0 - 127.0);
            //result.msNormal = normalize(modelFromTangent * tsNormal);
        }
        */

        // Encode transmission in baseColor's alpha
        /*float transmissionFactor = mat.transmissionFactor;
        if(mat.transmissionTexture >= 0)
        {
            uint txtId = mat.transmissionTexture;
            transmissionFactor *= texture(texturesMap[nonuniformEXT(txtId)], texcoord0).r;
        }
        result.baseColor.a = transmissionFactor;*/

        // Metallic & Roughness
        result.metallic = mat.metallic;
        result.roughness = mat.roughness*0.1;
        if(mat.pbrMetallicRoughnessTexture > -1)
        {
            uint txtId = mat.pbrMetallicRoughnessTexture;
            vec3 metallicRoughness = texture(texturesMap[nonuniformEXT(txtId)], texcoord0).xyz;
            result.metallic *= metallicRoughness.b;
            result.roughness *= metallicRoughness.g;
        }
    }
    else
    {
        result.baseColor = vec3(1);
        result.emittance = vec3(0);
        result.metallic = 0;
        result.roughness = 1;
    }

    if(albedo85)
    {
        result.baseColor = vec3(0.85);
    }

    return result;
}

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

  // Result
  outEmissive = vec4(surfaceProperties.emittance, 1);
  outNormals = vec4(surfaceProperties.msNormal, 1.0);
  outPBR = vec4(surfaceProperties.metallic, surfaceProperties.roughness, 0, 1);
  outBaseColor = vec4(surfaceProperties.baseColor, 1);
}
