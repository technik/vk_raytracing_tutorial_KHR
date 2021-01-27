// PBR functions
#include "gltf.glsl"

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

        if(mat.normalTexture >= 0)
        {
            // Tangent
            vec4 tan0 = getTangent(triangleIndex.x);
            vec4 tan1 = getTangent(triangleIndex.y);
            vec4 tan2 = getTangent(triangleIndex.z);

            vec3 msBitangent = cross(msNormal, msTangent.xyz) * msTangent.w;

            mat3 modelFromTangent = mat3(wsTangent, wsBitangent, worldNormal);

            uint txtId = mat.normalTexture;
            vec3 tsNormal = texture(texturesMap[nonuniformEXT(txtId)], texcoord0).xyz;
            tsNormal = pow(tsNormal, vec3(1/2.2)) + vec3(0,0,1e-2);
            tsNormal = normalize(tsNormal * 255.0 - 127.0);
            result.msNormal = normalize(modelFromTangent * tsNormal);
        }

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
}