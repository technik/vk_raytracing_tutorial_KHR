// Generate a random unsigned int from two unsigned int values, using 16 pairs
// of rounds of the Tiny Encryption Algorithm. See Zafar, Olano, and Curtis,
// "GPU Random Numbers via the Tiny Encryption Algorithm"

const float PI = 3.14159265;
#define M_PI PI
const float TwoPi = 6.2831852436065673828125f;

uint tea(uint val0, uint val1)
{
  uint v0 = val0;
  uint v1 = val1;
  uint s0 = 0;

  for(uint n = 0; n < 16; n++)
  {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

// Generate a random unsigned int in [0, 2^24) given the previous RNG state
// using the Numerical Recipes linear congruential generator
uint lcg(inout uint prev)
{
  uint LCG_A = 1664525u;
  uint LCG_C = 1013904223u;
  prev       = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// Generate a random float in [0, 1) given the previous RNG state
float rnd(inout uint prev)
{
  return (float(lcg(prev)) / float(0x01000000));
}

//-------------------------------------------------------------------------------------------------
// Sampling
//-------------------------------------------------------------------------------------------------

// Randomly sampling around +Z
vec3 samplingHemisphere(inout uint seed, in vec3 x, in vec3 y, in vec3 z)
{

  float r1 = rnd(seed);
  float r2 = rnd(seed);
  float sq = sqrt(1.0 - r2);

  vec3 direction = vec3(cos(2 * M_PI * r1) * sq, sin(2 * M_PI * r1) * sq, sqrt(r2));
  direction      = direction.x * x + direction.y * y + direction.z * z;

  return direction;
}

// Pixar's method for orthonormal basis generation
void createCoordinateSystem(in vec3 n, out vec3 b1, out vec3 b2)
{
  float sign = n.z > 0.0 ? 1.0 : -1.0;
  const float a = -1.0f / (sign + n.z);
  const float b = n.x * n.y * a;
  b1 = vec3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
  b2 = vec3(b, sign + n.y * n.y * a, -n.y);
}

vec3 randomUnitVector(in vec2 seed)
{
  float theta = TwoPi*seed.x;
  float z = 2*seed.y-1;
  float horRad = sqrt(1-z*z);
  return vec3(
  cos(theta)*horRad,
  z,
  sin(theta)*horRad
  );
}
