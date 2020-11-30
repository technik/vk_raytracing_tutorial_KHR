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
	prd.emittance = clearColor.xyz;
  	prd.world_position.w = -1.0;
}
