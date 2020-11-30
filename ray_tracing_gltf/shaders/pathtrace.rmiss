#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#include "raycommon.glsl"

layout(location = 0) rayPayloadInEXT hitPayload prd;

layout(push_constant) uniform Constants
{
  vec4 clearColor;
};

void main()
{
	const vec3 up = vec3(0.5, 0.7, 1.0);
	//prd.emittance = clearColor.xyz;
	prd.emittance = mix(up, vec3(1.0), gl_WorldRayDirectionEXT.y) * clearColor.xyz;
  	prd.world_position.w = -1.0;
}
