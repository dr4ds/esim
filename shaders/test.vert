#version 460

layout (location = 0) out vec4 out_col;

void main()
{
  vec4 v[3] = {
    vec4(-1, -1, 0.1, 1),
    vec4( 1,  1, 0.1, 1),
    vec4( 1, -1, 0.1, 1),
  };

  vec4 c[3] = {
    vec4(1,0,0,1),
    vec4(0,1,0,1),
    vec4(0,0,1,1),
  };

  gl_Position = v[gl_VertexIndex];
  out_col = c[gl_VertexIndex];
}
