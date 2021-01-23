
struct GltfShadeMaterial
{
  vec4 pbrBaseColorFactor;
  int  pbrBaseColorTexture;
  vec3 emissiveFactor;
  int  emissiveTexture;
  int  normalTexture;
  int   pbrMetallicRoughnessTexture;
  float roughness;
  float metallic;
  int padding[3];
};

#ifndef __cplusplus
struct PrimMeshInfo
{
  uint indexOffset;
  uint vertexOffset;
  int  materialIndex;
  uint numIndices;
};

struct LightInstanceInfo
{
  uint vtxOffset;
  uint indexOffset;
  uint numTriangles;
  uint matrixIndex;
  float weightedRadiance;
};

struct EmissiveTriangleInfo
{
  uint vtxOffset;
  uint indexOffset;
  uint matrixIndex;
  float weightedRadiance;
};

struct AliasTable
{
  uint Ki;
  float cutoff;
}

vec3 computeDiffuse(GltfShadeMaterial mat, vec3 lightDir, vec3 normal)
{
  // Lambertian
  float dotNL = max(dot(normal, lightDir), 0.0);
  return mat.pbrBaseColorFactor.xyz * dotNL;
}

vec3 computeSpecular(GltfShadeMaterial mat, vec3 viewDir, vec3 lightDir, vec3 normal)
{
  // Compute specular only if not in shadow
  const float kPi        = 3.14159265;
  const float kShininess = 60.0;

  // Specular
  const float kEnergyConservation = (2.0 + kShininess) / (2.0 * kPi);
  vec3        V                   = normalize(-viewDir);
  vec3        R                   = reflect(-lightDir, normal);
  float       specular            = kEnergyConservation * pow(max(dot(V, R), 0.0), kShininess);

  return vec3(specular);
}
#endif
