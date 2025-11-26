#version 460

layout (location = 0) in vec2 in_pos;

layout (location = 0) out vec4 out_col;

layout (std140, set = 1, binding = 0)
uniform UB {
  mat4 view;
  float pointsz;
} ub;

void main()
{
  gl_Position = ub.view * vec4(in_pos, 0, 1);
  gl_PointSize = ub.pointsz;
  out_col = vec4(0, 0, 0, 1);
}
