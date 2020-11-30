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
	vec3 rawColor = texture(noisyTxt, outUV).rgb;
	vec3 toneMapped = 1.0-exp(-rawColor);
	float gamma = 1. / 2.2;
	vec3 gammaCorrected = pow(toneMapped, vec3(gamma));
	fragColor = vec4(gammaCorrected, 1.0);
}
