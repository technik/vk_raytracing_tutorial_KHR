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
  	prd.world_position.w = -1.0;
}
