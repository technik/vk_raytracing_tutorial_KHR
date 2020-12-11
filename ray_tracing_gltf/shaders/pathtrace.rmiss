#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#include "raycommon.glsl"

layout(location = 0) rayPayloadInEXT hitPayload prd;

layout(push_constant) uniform Constants
{
  vec4 clearColor;
  vec3  lightPosition;
  float skyIntensity;
  float sunIntensity;
  int   frame;
  int   maxBounces;
  int   firstBounce;
  float focalDistance;
  float lensRadius;
  int   renderFlags;
};

void main()
{
	const vec3 up = vec3(0.5, 0.7, 1.0);
	prd.emittance = mix(up, vec3(1.0), gl_WorldRayDirectionEXT.y) * clearColor.xyz * skyIntensity;

	if((renderFlags & FLAG_GREY_FURNACE) > 0)
	{
		prd.emittance = vec3(0.7);
	}
  	prd.world_position.w = -1.0;
}
