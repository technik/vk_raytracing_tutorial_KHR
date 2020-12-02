
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