#version 460

layout (location = 0) in vec4 in_pos;
layout (location = 1) in vec4 in_col;

layout (location = 0) out vec4 out_col;

layout (std140, set = 1, binding = 0)
uniform UB {
  mat4 view;
} ub;

void main()
{
  gl_Position = ub.view * in_pos;
  out_col = in_col;
}
