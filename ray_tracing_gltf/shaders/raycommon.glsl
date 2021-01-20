struct hitPayload
{
	vec4 worldPos;
	vec3 worldNormal;
  	vec3 baseColor;
  	vec3 emittance;
  	float roughness;
  	float metallic;
  	uint seed;
};

#define FLAG_JITTER_AA	1
#define FLAG_DOF		2
#define FLAG_ALBEDO_85	4
#define FLAG_NO_SPEC	8
#define FLAG_NO_DIFF	16
