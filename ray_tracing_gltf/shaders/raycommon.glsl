
// Ray payloads

struct hitPayload
{
  vec4 world_position; // xyz: position, w: distance
  vec3 world_normal;
  vec3 emittance;
  vec4 baseColor;
  float roughness;
  float metallic;
};

// Render flags
#define FLAG_OVERRIDE_WHITE_DIFFUSE (1<<0)
#define FLAG_OVERRIDE_ALBEDO_85 (1<<1)
#define FLAG_OVERRIDE_MIRROR (1<<2)
#define FLAG_GREY_FURNACE (1<<3)
#define FLAG_DIFFUSE_ONLY (1<<4)
#define FLAG_SPECULAR_ONLY (1<<5)