#version 450
layout(location = 0) in vec2 outUV;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D noisyTxt;

layout(push_constant) uniform shaderInformation
{
  float aspectRatio;
}
pushc;

void main()
{
  vec2  uv    = outUV;
  float gamma = 1. / 2.2;
  vec3 s = texture(noisyTxt, uv).rgb;

  // Tone mapping
  float energy = sqrt(dot(s,s))/3;
  vec3 color = mix(s, vec3(energy), energy/(energy+1000*100));
  color = color / (color + 1);

  fragColor   = vec4(pow(color, vec3(gamma)), 1.0);
}
