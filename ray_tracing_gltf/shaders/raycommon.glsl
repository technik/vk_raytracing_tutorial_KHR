struct hitPayload
{
  vec4 world_position; // xyz: position, w: distance
  vec3 world_normal;
  vec3 emittance;
  vec4 baseColor;
  float roughness;
  float metallic;
};
