#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "mathx.h"

extern "C" {

#define shuffle(a, x, y, z, w) (__builtin_shufflevector(a, a, x, y, z, w))
#define shuffle2(a, b, x, y, z, w) (__builtin_shufflevector(a, b, x, y, z, w))

vec4 vec3to4(vec3 a)
{
  vec4 v = *(vec4 *)&a;
  v[3] = 0.0;

  return v;
}

vec3 vec4to3(vec4 a)
{
  return *(vec3 *)&a;
}

float rad(float deg)
{
  return deg*M_PI/180.0f;
}

ivec2 vec2_to_vec2i(vec2 a)
{
  return __builtin_convertvector(a, ivec2);
}

vec2 ivec2_to_vec2(ivec2 a)
{
  return __builtin_convertvector(a, vec2);
}

uvec2 vec2_to_vec2u(vec2 a)
{
  return __builtin_convertvector(a, uvec2);
}

vec2 uvec2_to_vec2(uvec2 a)
{
  return __builtin_convertvector(a, vec2);
}

uvec2 ivec2_to_vec2u(ivec2 a)
{
  return __builtin_convertvector(a, uvec2);
}

ivec2 uvec2_to_vec2i(uvec2 a)
{
  return __builtin_convertvector(a, ivec2);
}

ivec3 vec3_to_vec3i(vec3 a)
{
  return __builtin_convertvector(a, ivec3);
}

vec3  ivec3_to_vec3(ivec3 a)
{
  return __builtin_convertvector(a, vec3);
}

uvec3 vec3_to_vec3u(vec3 a)
{
  return __builtin_convertvector(a, uvec3);
}

vec3  uvec3_to_vec3(uvec3 a)
{
  return __builtin_convertvector(a, vec3);
}

uvec3 ivec3_to_vec3u(ivec3 a)
{
  return __builtin_convertvector(a, uvec3);
}

ivec3 uvec3_to_vec3i(uvec3 a)
{
  return __builtin_convertvector(a, ivec3);
}

ivec4 vec4_to_vec4i(vec4 a)
{
  return __builtin_convertvector(a, ivec4);
}

vec4  ivec4_to_vec4(ivec4 a)
{
  return __builtin_convertvector(a, vec4);
}

uvec4 vec4_to_vec4u(vec4 a)
{
  return __builtin_convertvector(a, uvec4);
}

vec4  uvec4_to_vec4(uvec4 a)
{
  return __builtin_convertvector(a, vec4);
}

uvec4 ivec4_to_vec4u(ivec4 a)
{
  return __builtin_convertvector(a, uvec4);
}

ivec4 uvec4_to_vec4i(uvec4 a)
{
  return __builtin_convertvector(a, ivec4);
}

ivec2 ivec2_init(int32_t x, int32_t y)
{
  return (ivec2){x, y};
}

ivec2 ivec2_zero(void)
{
  return (ivec2){0,0};
}

ivec2 ivec2_one(void)
{
  return (ivec2){1,1};
}


vec2 vec2_init(float x, float y)
{
  return (vec2){x, y};
}

vec2 vec2_zero(void)
{
  return vec2_init(0.0, 0.0);
}

vec2 point_one(void)
{
  return vec2_init(1.0, 1.0);
}

vec2 vec2_rotate(vec2 pos, vec2 origin, float angle)
{
  if (angle == 0.0)
    return pos;

  float sa = sinf(angle);
  float ca = cosf(angle);

  float x = pos[0]-origin[0];
  float y = pos[1]-origin[1];

  float rx = x * ca - y * sa;
  float ry = x * sa + y * ca;

  return vec2_init(rx + origin[0], ry + origin[1]);
}

float vec2_len(vec2 x)
{
  return sqrt(x[0]*x[0] + x[1]*x[1]);
}

vec2 vec2_norm(vec2 x)
{
  return x/vec2_len(x);
}

void vec2_print(vec2 x)
{
  printf("{ x: %f, y: %f }\n", x[0], x[1]);
}

vec3 vec3_init(float x, float y, float z)
{
  return (vec3){x, y, z};
}

vec3 vec3_initn(float n)
{
  return (vec3){n, n, n};
}

vec3 vec3_initv(vec4 a)
{
  return vec4to3(a);
}

vec3 vec3_initx(float x)
{
  return (vec3){x, 0, 0};
}

vec3 vec3_inity(float y)
{
  return (vec3){0, y, 0};
}

vec3 vec3_initz(float z)
{
  return (vec3){0, 0, z};
}

vec3 vec3_zero(void)
{
  return (vec3){0, 0, 0};
}

vec3 vec3_one(void)
{
  return (vec3){1, 1, 1};
}

float vec3_dot(vec3 a, vec3 b)
{
  vec3 v = a*b;

  return v[0]+v[1]+v[2];
}

vec3 vec3_cross(vec3 a, vec3 b)
{
  return vec3_init(
      a[1]*b[2] - a[2]*b[1],
      a[2]*b[0] - a[0]*b[2],
      a[0]*b[1] - a[1]*b[0]);
}

vec3 vec3_crossn(vec3 a, vec3 b)
{
  return vec3_norm(vec3_cross(a,b));
}

vec3 vec3_lerp(vec3 from, vec3 to, float t)
{
  vec3 s = vec3_initn(t);
  vec3 d = to - from;

  return from + (d*s);
}

float vec3_len(vec3 a)
{
  return sqrt(vec3_dot(a, a));
}

vec3 vec3_norm(vec3 a)
{
  return a/vec3_len(a);
}

bool vec3_eq(vec3 a, vec3 b)
{
  return a[0] == b[0] &&
         a[1] == b[1] &&
         a[2] == b[2];
}

void vec3_print(vec3 a)
{
  printf("{ %f %f %f }\n", a[0], a[1], a[2]);
}

vec4 vec4_init(float x, float y, float z, float w)
{
  return (vec4){x, y, z, w};
}

vec4 vec4_initn(float n)
{
  return (vec4){n,n,n,n};
}

vec4 vec4_initv(vec3 a, float w)
{
  vec4 r = vec3to4(a);
  r[3] = w;

  return r;
}

vec4 vec4_zero(void)
{
  return vec4_init(0.0, 0.0, 0.0, 0.0);
}

vec4 vec4_one(void)
{
  return vec4_init(1.0, 1.0, 1.0, 1.0);
}

void vec4_print(vec4 x)
{
  printf("{ %f %f %f %f }\n", x[0], x[1], x[2], x[3]);
}

vec4 vec4_splat_x(vec4 a)
{
  return shuffle(a, 0, 0, 0, 0);
}

vec4 vec4_splat_y(vec4 a)
{
  return shuffle(a, 1, 1, 1, 1);
}

vec4 vec4_splat_z(vec4 a)
{
  return shuffle(a, 2, 2, 2, 2);
}

vec4 vec4_splat_w(vec4 a)
{
  return shuffle(a, 3, 3, 3, 3);
}

float vec4_dot(vec4 a, vec4 b)
{
  vec4 c = a*b;

  return c[0]+c[1]+c[2]+c[3];
}

float vec4_len(vec4 a)
{
  return sqrt(vec4_dot(a, a));
}

vec4 vec4_norm(vec4 a)
{
  return a/vec4_len(a);
}

vec4 vec4_cross(vec4 a, vec4 b)
{
  vec4 t0 = shuffle(a,3,0,2,1);
  vec4 t2 = t0*b;

  return (t0*shuffle(b,3,1,0,2))-shuffle(t2,3,0,2,1);
}

vec4 vec4_crossn(vec4 a, vec4 b)
{
  return vec4_norm(vec4_cross(a, b));
}

mat4 mmul(mat4 a, mat4 b)
{
  vec4 c0 = b.data[0];
  vec4 c1 = b.data[1];
  vec4 c2 = b.data[2];
  vec4 c3 = b.data[3];

  vec4 x = a.data[0];
  vec4 v0 = vec4_splat_x(c0) * x;
  vec4 v1 = vec4_splat_x(c1) * x;
  vec4 v2 = vec4_splat_x(c2) * x;
  vec4 v3 = vec4_splat_x(c3) * x;

  x = a.data[1];
  v0 += vec4_splat_y(c0) * x;
  v1 += vec4_splat_y(c1) * x;
  v2 += vec4_splat_y(c2) * x;
  v3 += vec4_splat_y(c3) * x;

  x = a.data[2];
  v0 += vec4_splat_z(c0) * x;
  v1 += vec4_splat_z(c1) * x;
  v2 += vec4_splat_z(c2) * x;
  v3 += vec4_splat_z(c3) * x;

  x = a.data[3];
  v0 += vec4_splat_w(c0) * x;
  v1 += vec4_splat_w(c1) * x;
  v2 += vec4_splat_w(c2) * x;
  v3 += vec4_splat_w(c3) * x;

  return (mat4){v0, v1, v2, v3};
}

mat4 transpose(mat4 a)
{
  vec4 r0 = a.data[0];
  vec4 r1 = a.data[1];
  vec4 r2 = a.data[2];
  vec4 r3 = a.data[3];

  vec4 tmp0 = shuffle2(r0, r1, 0, 4, 1, 5);
  vec4 tmp1 = shuffle2(r0, r1, 2, 6, 3, 7);
  vec4 tmp2 = shuffle2(r2, r3, 0, 4, 1, 5);
  vec4 tmp3 = shuffle2(r2, r3, 2, 6, 3, 7);

  return (mat4){{
    shuffle2(tmp0, tmp2, 0, 1, 4, 5),
    shuffle2(tmp0, tmp2, 2, 3, 6, 7),
    shuffle2(tmp1, tmp3, 0, 1, 4, 5),
    shuffle2(tmp1, tmp3, 2, 3, 6, 7),
  }};
}

mat4 invert(mat4 a)
{
  vec4 c0 = a.data[0];
  vec4 c1 = a.data[1];;
  vec4 c2 = a.data[2];;
  vec4 c3 = a.data[3];;

  vec4 tmp, r0, r1, r2, r3;

  tmp = shuffle2(c0, c2, 0, 4, 1, 5);
  r1  = shuffle2(c1, c3, 0, 4, 1, 5);

  r0  = shuffle2(tmp, r1, 0, 4, 1, 5);
  r1  = shuffle2(tmp, r1, 2, 6, 3, 7);

  tmp = shuffle2(c0, c2, 2, 6, 3, 7);
  r3  = shuffle2(c1, c3, 2, 6, 3, 7);

  r2  = shuffle2(tmp, r3, 0, 4, 1, 5);
  r3  = shuffle2(tmp, r3, 2, 6, 3, 7);

  r1  = shuffle(r1, 2, 3, 0, 1);
  r3  = shuffle(r3, 2, 3, 0, 1);

  tmp = r2 * r3;
  tmp = shuffle(tmp, 1, 0, 7, 6);

  c0  = r1 * tmp;
  c1  = r0 * tmp;

  tmp = shuffle(tmp, 2, 3, 4, 5);

  c0  = r1 * tmp - c0;
  c1  = r0 * tmp - c1;
  c1  = shuffle(c1, 2, 3, 4, 5);

  tmp = r1 * r2;
  tmp = shuffle(tmp, 1, 0, 7, 6);

  c0  = r3 * tmp + c0;
  c3  = r0 * tmp;

  tmp = shuffle(tmp, 2, 3, 4, 5);

  c0  = c0 - r3 * tmp;
  c3  = r0 * tmp - c3;
  c3  = shuffle(c3, 2, 3, 4, 5);

  tmp = shuffle(r1, 2, 3, 4, 5) * r3;
  tmp = shuffle(tmp, 1, 0, 7, 6);
  r2  = shuffle(r2, 2, 3, 4, 5);

  c0  = r2 * tmp + c0;
  c2  = r0 * tmp;

  tmp = shuffle(tmp, 2, 3, 4, 5);

  c0  = c0 - r2 * tmp;
  c2  = r0 * tmp - c2;
  c2  = shuffle(c2, 2, 3, 4, 5);

  tmp = r0 * r1;
  tmp = shuffle(tmp, 1, 0, 7, 6);

  c2  = r3 * tmp + c2;
  c3  = r2 * tmp - c3;

  tmp = shuffle(tmp, 2, 3, 4, 5);

  c2  = r3 * tmp - c2;
  c3  = c3 - r2 * tmp;

  tmp = r0 * r3;
  tmp = shuffle(tmp, 1, 0, 7, 6);

  c1  = c1 - r2 * tmp;
  c2  = r1 * tmp + c2;

  tmp = shuffle(tmp, 2, 3, 4, 5);

  c1  = r2 * tmp + c1;
  c2  = c2 - r1 * tmp;

  tmp = r0 * r2;
  tmp = shuffle(tmp, 1, 0, 7, 6);

  c1  = r3 * tmp + c1;
  c3  = c3 - r1 * tmp;

  tmp = shuffle(tmp, 2, 3, 4, 5);

  c1  = c1 - r3 * tmp;
  c3  = r1 * tmp + c3;

  vec4 det = r0 * c0;
  det = shuffle(det, 2, 3, 4, 5) + det;
  det = shuffle(det, 1, 0, 7, 6) + det;

  det = 1.0f / det;

  return (mat4){c0 * det, c1 * det, c2 * det, c3 * det};
}

mat4 mat_identity(void)
{
  return (mat4){{
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0},
    {0.0, 0.0, 0.0, 1.0},
  }};
}

mat4 mat_zero(void)
{
  return (mat4){{
    {0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0},
  }};
}

mat4 mat_one(void)
{
  return (mat4){{
    {1.0, 1.0, 1.0, 1.0},
    {1.0, 1.0, 1.0, 1.0},
    {1.0, 1.0, 1.0, 1.0},
    {1.0, 1.0, 1.0, 1.0},
  }};
}

mat4 translation(vec3 x)
{
  return (mat4){{
    { 1.0,  0.0,  0.0,  0.0},
    { 0.0,  1.0,  0.0,  0.0},
    { 0.0,  0.0,  1.0,  0.0},
    {x[0], x[1], x[2],  1.0},
  }};
}

mat4 scaling(vec3 x)
{
  return (mat4){{
    {x[0],  0.0,  0.0, 0.0},
    { 0.0, x[1],  0.0, 0.0},
    { 0.0,  0.0, x[2], 0.0},
    { 0.0,  0.0,  0.0, 1.0},
  }};
}

mat4 rotation_x(float angle)
{
  float ca = cosf(angle);
  float sa = sinf(angle);

  return (mat4){{
    { 1.0,  0.0,  0.0, 0.0},
    { 0.0,   ca,   sa, 0.0},
    { 0.0,  -sa,   ca, 0.0},
    { 0.0,  0.0,  0.0, 1.0},
  }};
}

mat4 rotation_y(float angle)
{
  float ca = cosf(angle);
  float sa = sinf(angle);

  return (mat4){{
    {  ca,  0.0,  -sa, 0.0},
    { 0.0,  1.0,  0.0, 0.0},
    {  sa,  0.0,   ca, 0.0},
    { 0.0,  0.0,  0.0, 1.0},
  }};
}

mat4 orthographic_lhzo(float left, float right, float bottom, float top,
                       float near, float far)
{
  mat4 rm = mat_zero();
  vec4 *d = rm.data;

  float rl =  1.f / (right - left);
  float tb =  1.f / (top - bottom);
  float fn = -1.f / (far - near);

  d[0][0] = 2.f * rl;
  d[1][1] = 2.f * tb;
  d[2][2] = -fn;
  d[3][0] = -(right  + left)    * rl;
  d[3][1] = -(top    + bottom)  * tb;
  d[3][2] = near * fn;
  d[3][3] = 1.0f;

  return rm;
}

mat4 orthographic_lhno(float left, float right, float bottom, float top,
                       float near, float far)
{
  mat4 rm = mat_zero();
  vec4 *d = rm.data;

  float rl =  1.f / (right - left);
  float tb =  1.f / (top - bottom);
  float fn = -1.f / (far - near);

  d[0][0] = 2.f * rl;
  d[1][1] = 2.f * tb;
  d[2][2] = -2.f * -fn;
  d[3][0] = -(right  + left)    * rl;
  d[3][1] = -(top    + bottom)  * tb;
  d[3][2] = (far + near) * fn;
  d[3][3] = 1.0f;

  return rm;
}

mat4 perspective_lhzo(float aspect, float fov, float near, float far)
{
  mat4 rm = mat_zero();
  vec4 *d = rm.data;

  float f  = 1.0f / tanf(fov * 0.5f);
  float fn = 1.0f / (near - far);

  d[0][0] = f / aspect;
  d[1][1] = f;
  d[2][2] = -far * fn;
  d[2][3] = 1.0f;
  d[3][2] = near * far * fn;

  return rm;
}

mat4 perspective_lhno(float aspect, float fov, float near, float far)
{
  mat4 rm = mat_zero();
  vec4 *d = rm.data;

  float f  = 1.0f / tanf(fov * 0.5f);
  float fn = 1.0f / (near - far);

  d[0][0] = f / aspect;
  d[1][1] = f;
  d[2][2] = -(near + far) * fn;
  d[2][3] = 1.0f;
  d[3][2] = 2.0f * near * far * fn;

  return rm;
}

mat4 perspective_rhzo(float aspect, float fov, float near, float far)
{
  mat4 rm = mat_zero();
  vec4 *d = rm.data;

  float f  = 1.0f / tanf(fov * 0.5f);
  float fn = 1.0f / (near - far);

  d[0][0] = f / aspect;
  d[1][1] = f;
  d[2][2] = far * fn;
  d[2][3] = -1.0f;
  d[3][2] = near * far * fn;

  return rm;
}

mat4 perspective_rhno(float aspect, float fov, float near, float far)
{
  mat4 rm = mat_zero();
  vec4 *d = rm.data;

  float f  = 1.0f / tanf(fov * 0.5f);
  float fn = 1.0f / (near - far);

  d[0][0] = f / aspect;
  d[1][1] = f;
  d[2][2] = (near + far) * fn;
  d[2][3] = -1.0f;
  d[3][2] = 2.0f * near * far * fn;

  return rm;
}

mat4 look_at(vec3 eye, vec3 center, vec3 up)
{
  vec3 f = vec3_norm(center - eye);
  // vprint(f);
  vec3 s = vec3_crossn(up, f);
  // vprint(s);
  vec3 u = vec3_cross(f, s);

  assert(0);

  mat4 rm = mat_zero();
  vec4 *d = rm.data;

  d[0][0] = s[0];
  d[0][1] = u[0];
  d[0][2] = f[0];
  d[1][0] = s[1];
  d[1][1] = u[1];
  d[1][2] = f[1];
  d[2][0] = s[2];
  d[2][1] = u[2];
  d[2][2] = f[2];
  d[3][0] = -vec3_dot(s, eye);
  d[3][1] = -vec3_dot(u, eye);
  d[3][2] = -vec3_dot(f, eye);
  d[3][3] = 1.0f;

  return rm;
}

vec3 mmulv3(mat4 a, vec3 b)
{
  vec4 r = mmulv4(a, vec4_initv(b, 1.0));

  return vec4to3(r);
}

vec4 mmulv4(mat4 a, vec4 b)
{
  vec4 *d = a.data;

  vec4 r;
  r  = d[0]*b[0];
  r += d[1]*b[1];
  r += d[2]*b[2];
  r += d[3]*b[3];

  return r;
}

void mprint(mat4 x)
{
  vec4 *d = x.data;

  printf("column-major 4x4 {\n %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n %f %f %f %f }\n",
         d[0][0], d[1][0], d[2][0], d[3][0],
         d[0][1], d[1][1], d[2][1], d[3][1],
         d[0][2], d[1][2], d[2][2], d[3][2],
         d[0][3], d[1][3], d[2][3], d[3][3]);

  printf("column-major 4x4 memory { %f %f %f %f | %f %f %f %f | %f %f %f %f | %f %f %f %f }\n",
         d[0][0], d[0][1], d[0][2], d[0][3], 
         d[1][0], d[1][1], d[1][2], d[1][3], 
         d[2][0], d[2][1], d[2][2], d[2][3], 
         d[3][0], d[3][1], d[3][2], d[3][3]);
}

quat quat_identity(void)
{
  return (quat){0,0,0,1};
}

quat quat_initv(vec3 a, float angle)
{
  float ha = angle*0.5;
  float s = sinf(ha);

  a = vec3_norm(a);

  return (quat){a[0]*s, a[1]*s, a[2]*s, cosf(ha)};
}

quat quat_init(float x, float y, float z, float angle)
{
  return quat_initv(vec3_init(x, y, z), angle);
}

quat quat_init_deg(float x, float y, float z)
{
  if (x+y+z == 0.0)
    return quat_identity();

  float len = sqrt(x*x+y*y+z*z);
  float angle = rad(len);

  return quat_init(x/len, y/len, z/len, angle);
}

quat qmul(quat a, quat b)
{
#if 0
  return (quat){
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
  };
#else
  vec4 negative = (vec4){-0.0,0.0,0.0,0.0};

  vec4 x = shuffle(negative,1,1,1,0);
  vec4 y = shuffle(a,0,1,2,0) * shuffle(b,3,3,3,0) +
           shuffle(a,1,2,0,1) * shuffle(b,2,0,1,1);

/*
  vec4 xr = (vec4)((i32x4)x ^ (i32x4)y);
*/
  vec4 xr = ivec4_to_vec4(vec4_to_vec4i(x) ^ vec4_to_vec4i(y));

  return shuffle(a,3,3,3,3) * b + xr -
         shuffle(a,2,0,1,2) * shuffle(b,1,2,0,2);
#endif
}

mat4 quat_to_mat(quat q)
{
  mat4 a = (mat4){{
    { q[3],  q[2], -q[1],  q[0]},
    {-q[2],  q[3],  q[0],  q[1]},
    { q[1], -q[0],  q[3],  q[2]},
    {-q[0], -q[1], -q[2],  q[3]},
  }};

  mat4 b = (mat4){{
    { q[3],  q[2], -q[1], -q[0]},
    {-q[2],  q[3],  q[0], -q[1]},
    { q[1], -q[0],  q[3], -q[2]},
    { q[0],  q[1],  q[2],  q[3]},
  }};

  return mmul(a, b);
}

vec3 rotate(quat a, vec3 b)
{
  vec3 im = vec4to3(a);

  vec3 v0 = im * (vec3_dot(im, b) * 2.0f);
  v0 += b * (a[3]*a[3] - vec3_dot(im, im));

  vec3 v1 = vec3_cross(im, b) * (2.0f*a[3]);

  return v0 + v1;
}

mat4 quat_look(vec3 eye, quat ori)
{
  mat4 m0 = transpose(quat_to_mat(ori));
  m0.data[3] = vec4_initv(-mmulv3(m0, eye), 1.0);

  return m0;
}

void qprint(quat x)
{
  printf("{ { %f %f %f }, %f }\n", x[0], x[1], x[2], x[3]);
}

#include <time.h>
#include <stdlib.h>

static float rand_float(void)
{
  return ((float)(rand() / (RAND_MAX + 1.0)));
}

static mat4 rand_mat(void)
{
  vec4 a = vec4_init(rand_float(), rand_float(), rand_float(), rand_float());
  vec4 b = vec4_init(rand_float(), rand_float(), rand_float(), rand_float());
  vec4 c = vec4_init(rand_float(), rand_float(), rand_float(), rand_float());
  vec4 d = vec4_init(rand_float(), rand_float(), rand_float(), rand_float());

  return (mat4){a, b, c, d};
}

static bool mat_eq(mat4 a, mat4 b)
{
  for (size_t i = 0; i < 4; i++)
    for (size_t j = 0; j < 4; j++)
      if (a.data[i][j] != b.data[i][j])
        return false;

  return true;
}

static void test_mmul(void)
{
  size_t cnt = 10000;
  while (cnt--) {
    mat4 a = rand_mat();
    mat4 b = rand_mat();
    mat4 c = mat_zero();

    for (size_t i = 0; i < 4; i++) {
      for (size_t j = 0; j < 4; j++) {
        for (size_t k = 0; k < 4; k++)
          c.data[i][j] += a.data[k][j] * b.data[i][k];
      }
    }

    mat4 d = mmul(a, b);

    if (!mat_eq(c, d)) {
      mprint(a);
      mprint(b);
      mprint(c);
      mprint(d);
      fprintf(stderr, "failed mmul %zu\n", cnt);
      exit(1);
    }
  }
}

static void test_transpose(void)
{
  size_t cnt = 10000;
  while (cnt--) {
    mat4 a = rand_mat();
    mat4 b = transpose(a);
    mat4 c = mat_zero();

    for (size_t i = 0; i < 4; i++)
      for (size_t j = 0; j < 4; j++)
        c.data[i][j] = a.data[j][i];

    if (!mat_eq(b, c)) {
      mprint(a);
      mprint(b);
      mprint(c);
      fprintf(stderr, "failed transpose %zu\n", cnt);
      exit(1);
    }
  }
}

void test_mathx(void)
{
  srand((unsigned int)time(NULL));

  test_mmul();
  test_transpose();
}

}
