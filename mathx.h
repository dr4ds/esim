#ifndef MATHX_H
#define MATHX_H

extern "C" {

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include <stdint.h>

#ifndef max
#define max(x, y) ((x > y) ? x : y)
#endif

#ifndef min
#define min(x, y) ((x > y) ? y : x)
#endif

#ifndef clamp
#define clamp(x, l, h) (x > h ? h : x < l ? l : x)
#endif

typedef float vec2 __attribute((vector_size(8), aligned(8)));
typedef float vec3 __attribute((vector_size(12), aligned(16)));
typedef float vec4 __attribute((vector_size(16), aligned(16)));

typedef int32_t ivec2 __attribute((vector_size(8), aligned(8)));
typedef int32_t ivec3 __attribute((vector_size(12), aligned(16)));
typedef int32_t ivec4 __attribute((vector_size(16), aligned(16)));

typedef uint32_t uvec2 __attribute((vector_size(8), aligned(8)));
typedef uint32_t uvec3 __attribute((vector_size(12), aligned(16)));
typedef uint32_t uvec4 __attribute((vector_size(16), aligned(16)));

typedef vec4 quat;

vec4 vec3to4(vec3 a);
vec3 vec4to3(vec4 a);

ivec2 vec2_to_vec2i(vec2 a);
vec2  ivec2_to_vec2(ivec2 a);
uvec2 vec2_to_vec2u(vec2 a);
vec2  uvec2_to_vec2(uvec2 a);
uvec2 ivec2_to_vec2u(ivec2 a);
ivec2 uvec2_to_vec2i(uvec2 a);

ivec3 vec3_to_vec3i(vec3 a);
vec3  ivec3_to_vec3(ivec3 a);
uvec3 vec3_to_vec3u(vec3 a);
vec3  uvec3_to_vec3(uvec3 a);
uvec3 ivec3_to_vec3u(ivec3 a);
ivec3 uvec3_to_vec3i(uvec3 a);

ivec4 vec4_to_vec4i(vec4 a);
vec4  ivec4_to_vec4(ivec4 a);
uvec4 vec4_to_vec4u(vec4 a);
vec4  uvec4_to_vec4(uvec4 a);
uvec4 ivec4_to_vec4u(ivec4 a);
ivec4 uvec4_to_vec4i(uvec4 a);

struct mat4 {
  vec4 data[4]; /* column major */
};

typedef struct mat4 mat4;

float rad(float deg);

ivec2 ivec2_init(int32_t x, int32_t y);
ivec2 ivec2_zero(void);
ivec2 ivec2_one(void);

vec2 vec2_init(float x, float y);
vec2 vec2_one(void);
vec2 vec2_zero(void);
vec2 vec2_rotate(vec2 a, vec2 origin, float angle);
float vec2_len(vec2 a);
vec2 vec2_norm(vec2 a);
void vec2_print(vec2 a);

vec3 vec3_init(float x, float y, float z);
vec3 vec3_initn(float n);
vec3 vec3_initv(vec4 a);
vec3 vec3_initx(float x);
vec3 vec3_inity(float y);
vec3 vec3_initz(float z);
vec3 vec3_zero(void);
vec3 vec3_one(void);
float vec3_dot(vec3 a, vec3 b);
float vec3_len(vec3 a);
vec3 vec3_norm(vec3 a);
vec3 vec3_cross(vec3 a, vec3 b);
vec3 vec3_crossn(vec3 a, vec3 b);
vec3 vec3_lerp(vec3 from, vec3 to, float t);
bool vec3_eq(vec3 a, vec3 b);
void vec3_print(vec3 a);

vec4 vec4_init(float x, float y, float z, float w);
vec4 vec4_initn(float n);
vec4 vec4_initv(vec3 a, float w);
vec4 vec4_zero(void);
vec4 vec4_one(void);
float vec4_dot(vec4 a, vec4 b);
float vec4_len(vec4 a);
vec4 vec4_norm(vec4 a);
vec4 vec4_cross(vec4 a, vec4 b);
vec4 vec4_crossn(vec4 a, vec4 b);
vec4 vec4_splat_x(vec4 a);
vec4 vec4_splat_y(vec4 a);
vec4 vec4_splat_z(vec4 a);
vec4 vec4_splat_w(vec4 a);
void vec4_print(vec4 x);

#define dot(x, y) _Generic((x), \
    vec3: vec3_dot, \
    vec4: vec4_dot  \
    )(x, y)

#define cross(x, y) _Generic((x), \
    vec3: vec3_cross, \
    vec4: vec4_cross  \
    )(x, y)

#define crossn(x, y) _Generic((x), \
    vec3: vec3_crossn, \
    vec4: vec4_crossn  \
    )(x, y)

#define normalize(x) _Generic((x), \
    vec2: vec2_norm, \
    vec3: vec3_norm, \
    vec4: vec4_norm  \
    )(x)

#define length(x) _Generic((x), \
    vec2: vec2_len, \
    vec3: vec3_len, \
    vec4: vec4_len  \
    )(x)

#define vprint(x) _Generic((x), \
    vec2: vec2_print, \
    vec3: vec3_print, \
    vec4: vec4_print  \
    )(x)

mat4 mat_identity(void);
mat4 mat_zero(void);
mat4 mat_one(void);
mat4 transpose(mat4 x);
mat4 invert(mat4 a);
mat4 translation(vec3 x);
mat4 scaling(vec3 x);
mat4 rotation_x(float angle);
mat4 rotation_y(float angle);
mat4 orthographic_lhzo(float left, float right, float bottom, float top,
                       float near, float far);
mat4 orthographic_lhno(float left, float right, float bottom, float top,
                       float near, float far);
mat4 perspective_lhzo(float ratio, float fov, float near, float far);
mat4 perspective_lhno(float ratio, float fov, float near, float far);
mat4 perspective_rhzo(float ratio, float fov, float near, float far);
mat4 perspective_rhno(float ratio, float fov, float near, float far);
// mat4 look_at(vec3 eye, vec3 center, vec3 up);
mat4 mmul(mat4 x, mat4 y);
vec3 mmulv3(mat4 a, vec3 b);
vec4 mmulv4(mat4 a, vec4 b);
void mprint(mat4 x);

quat quat_identity(void);
quat quat_initv(vec3 x, float angle);
quat quat_init(float x, float y, float z, float angle);
quat quat_init_deg(float x, float y, float z);
quat qmul(quat a, quat b);
mat4 quat_to_mat(quat q);
vec3 rotate(quat a, vec3 b);
// mat4 quat_look(vec3 eye, quat ori);
void qprint(quat x);

void test_mathx(void);

}

#endif
