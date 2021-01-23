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
#define FLAG_NEXT_EE	32
#define FLAG_EMIS_TRIS	64
#define FLAG_USE_ALIAS	128
