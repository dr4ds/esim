#include "imgui.h"
#include "implot.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlgpu3.h"
#include "mathx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <SDL3/SDL.h>

#define arraylen(x) (sizeof(x)/sizeof(*x))

struct Shape;

struct Model {
  const char *name;
  Shape *shapes;
  size_t shape_cnt;
  vec4 bbox;
};

enum class ShapeKind {
  None,
  Circle,
  Triangle,
  Rectangle,
  Bezier,
  Model
};

struct Shape {
  ShapeKind kind;

  union {
    struct {
      vec2 pos;
      float r;
    } circle;

    struct {
      vec2 a;
      vec2 b;
      vec2 c;
    } tri;

    struct {
      vec2 a;
      vec2 b;
      float w;
    } rect;

    Model mod;
  };
};

template <typename T>
struct Buffer {
  T *data;
  size_t pos;
  size_t cap;
};

template <typename T>
void buffer_init(Buffer<T> *b, size_t cap = 16)
{
  b->data = new T[cap];
  b->pos = 0;
  b->cap = cap;
}

template <typename T>
T *buffer_next(Buffer<T> *b)
{
  if (b->pos >= b->cap) {
    size_t ncap = b->cap*2;
    T *ndata = new T[ncap];
    memcpy(ndata, b->data, sizeof(T)*b->pos);
    delete[] b->data;
    b->data = ndata;
    b->cap = ncap;
  }

  T *res = b->data + b->pos;
  b->pos++;

  return res;
}

template <typename T>
T *buffer_at(Buffer<T> *b, size_t idx)
{
  return &b->data[idx];
}

template <typename T>
void buffer_deinit(Buffer<T> *b)
{
  delete[] b->data;
}

template <typename T>
void buffer_reset(Buffer<T> *b)
{
  b->pos = 0;
}

struct PointVertex {
  vec2 pos;
};

struct Point {
  SDL_GPUGraphicsPipeline *pipeline;
  SDL_GPUBuffer *vbuf;
  SDL_GPUTransferBuffer *vtbuf;
  Buffer<PointVertex> vertices;
};

struct FillVertex {
  vec4 pos;
  vec4 col;
};

struct Fill {
  SDL_GPUGraphicsPipeline *pipeline;
  SDL_GPUBuffer *vbuf;
  SDL_GPUTransferBuffer *vtbuf;
  Buffer<FillVertex> vertices;
};

struct Grid {
  int width;
  int height;
  float tilesz;
  float size;
};

enum class CursorState {
  Move,
  PlaceShape,
  PlaceModel,
};

struct Cursor {
  CursorState state;
  vec2 raw;
  vec2 pos;
  vec2 grid;
};

struct Camera {
  vec3 pos;
  vec3 target;
  float scale;
};

struct State {
  float scale;
  SDL_Window *win;
  SDL_GPUDevice *dev;
  SDL_GPUTexture *swapchain_texture;
  SDL_GPUTextureFormat sc_format;
  SDL_GPUSampleCount sample_cnt;
  SDL_GPUTexture *render_txt;
  SDL_GPUTexture *resolve_txt;
  int w;
  int h;

  Fill fill;
  Point point;
  SDL_GPUGraphicsPipeline *test_pipeline;
  Grid grid;
  Cursor cursor;
  Camera cam;
  uint64_t start_time; // program start time
  uint64_t elapsed_time;
  uint64_t prev_time;  // previous frame start time
  uint64_t curr_time;  // current frame start time
  uint64_t frame;
  mat4 proj;
  mat4 view;
  const bool *kb;

  Shape new_shape;
  size_t new_shape_vertex_cnt;
  Buffer<Shape> shapes;
  size_t selidx;

  bool en_drag;
  bool en_settings_window;
};
static State state;

void init_fill(void)
{
  // init buffers
  size_t scnt = 1<<16;
  uint32_t vsz = sizeof(FillVertex)*scnt;
  buffer_init(&state.fill.vertices, scnt);

  SDL_GPUBufferCreateInfo buffer_info = {
    .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
    .size = vsz,
  };

  state.fill.vbuf = SDL_CreateGPUBuffer(state.dev, &buffer_info);
  if (!state.fill.vbuf)
    abort();

  SDL_GPUTransferBufferCreateInfo tbuffer_info = {
    .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
    .size = vsz,
  };

  state.fill.vtbuf = SDL_CreateGPUTransferBuffer(state.dev,
      &tbuffer_info);
  if (!state.fill.vtbuf)
    abort();

  // init shaders
  size_t vsrc_sz; 
  void* vsrc = SDL_LoadFile("shaders/fill.vert.spv", &vsrc_sz);

  SDL_GPUShaderCreateInfo vinfo = {
    .code_size = vsrc_sz,
    .code = (uint8_t *)vsrc,
    .entrypoint = "main",
    .format = SDL_GPU_SHADERFORMAT_SPIRV,
    .stage = SDL_GPU_SHADERSTAGE_VERTEX,
    .num_uniform_buffers = 1,
  };
  SDL_GPUShader *vshader = SDL_CreateGPUShader(state.dev, &vinfo);
  SDL_free(vsrc);

  if (!vshader)
    abort();

  size_t fsrc_sz; 
  void* fsrc = SDL_LoadFile("shaders/fill.frag.spv", &fsrc_sz);

  SDL_GPUShaderCreateInfo finfo = {
    .code_size = fsrc_sz,
    .code = (uint8_t *)fsrc,
    .entrypoint = "main",
    .format = SDL_GPU_SHADERFORMAT_SPIRV,
    .stage = SDL_GPU_SHADERSTAGE_FRAGMENT,
  };
  SDL_GPUShader *fshader = SDL_CreateGPUShader(state.dev, &finfo);
  SDL_free(fsrc);

  if (!fshader)
    abort();

  // pipeline
  SDL_GPUVertexBufferDescription vbdesc[] = {
    {
      .slot = 0,
      .pitch = sizeof(FillVertex),
      .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
    },
  };

  SDL_GPUVertexAttribute vadesc[] = {
    {
      .location = 0,
      .buffer_slot = 0,
      .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4,
      .offset = offsetof(FillVertex, pos),
    },
    {
      .location = 1,
      .buffer_slot = 0,
      .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4,
      .offset = offsetof(FillVertex, col),
    },
  };

  SDL_GPUVertexInputState vinput = {
    .vertex_buffer_descriptions = vbdesc,
    .num_vertex_buffers = arraylen(vbdesc),
    .vertex_attributes = vadesc,
    .num_vertex_attributes = arraylen(vadesc)
  };

  SDL_GPURasterizerState raster = {
    .cull_mode = SDL_GPU_CULLMODE_NONE,
  };

  SDL_GPUColorTargetDescription color_desc[] = {
    {
      .format = SDL_GetGPUSwapchainTextureFormat(state.dev, state.win),
      .blend_state = {
        .src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
        .dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
        .color_blend_op = SDL_GPU_BLENDOP_ADD,
        .src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
        .dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
        .alpha_blend_op = SDL_GPU_BLENDOP_ADD,
        .enable_blend = true,
      },
    },
  };

  SDL_GPUGraphicsPipelineTargetInfo target_info = {
    .color_target_descriptions = color_desc,
    .num_color_targets = arraylen(color_desc),
  };

  SDL_GPUGraphicsPipelineCreateInfo fill_pipeline_info = {
    .vertex_shader = vshader,
    .fragment_shader = fshader,
    .vertex_input_state = vinput,
    .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
    .rasterizer_state = raster,
    .multisample_state = {.sample_count = state.sample_cnt},
    .target_info = target_info,
  };

  state.fill.pipeline = SDL_CreateGPUGraphicsPipeline(state.dev,
      &fill_pipeline_info);
  if (!state.fill.pipeline)
    abort();

  SDL_ReleaseGPUShader(state.dev, vshader);
  SDL_ReleaseGPUShader(state.dev, fshader);
}

void init_point(void)
{
  // init buffers
  size_t scnt = 1<<21; // 2MB
  uint32_t vsz = sizeof(PointVertex)*scnt;
  buffer_init(&state.point.vertices, scnt);

  SDL_GPUBufferCreateInfo buffer_info = {
    .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
    .size = vsz,
  };

  state.point.vbuf = SDL_CreateGPUBuffer(state.dev, &buffer_info);
  if (!state.point.vbuf)
    abort();

  SDL_GPUTransferBufferCreateInfo tbuffer_info = {
    .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
    .size = vsz,
  };

  state.point.vtbuf = SDL_CreateGPUTransferBuffer(state.dev,
      &tbuffer_info);
  if (!state.point.vtbuf)
    abort();

  // init shaders
  size_t vsrc_sz; 
  void* vsrc = SDL_LoadFile("shaders/point.vert.spv", &vsrc_sz);

  SDL_GPUShaderCreateInfo vinfo = {
    .code_size = vsrc_sz,
    .code = (uint8_t *)vsrc,
    .entrypoint = "main",
    .format = SDL_GPU_SHADERFORMAT_SPIRV,
    .stage = SDL_GPU_SHADERSTAGE_VERTEX,
    .num_uniform_buffers = 1,
  };
  SDL_GPUShader *vshader = SDL_CreateGPUShader(state.dev, &vinfo);
  SDL_free(vsrc);

  if (!vshader)
    abort();

  size_t fsrc_sz; 
  void* fsrc = SDL_LoadFile("shaders/point.frag.spv", &fsrc_sz);

  SDL_GPUShaderCreateInfo finfo = {
    .code_size = fsrc_sz,
    .code = (uint8_t *)fsrc,
    .entrypoint = "main",
    .format = SDL_GPU_SHADERFORMAT_SPIRV,
    .stage = SDL_GPU_SHADERSTAGE_FRAGMENT,
  };
  SDL_GPUShader *fshader = SDL_CreateGPUShader(state.dev, &finfo);
  SDL_free(fsrc);

  if (!fshader)
    abort();

  // pipeline
  SDL_GPUVertexBufferDescription vbdesc[] = {
    {
      .slot = 0,
      .pitch = sizeof(PointVertex),
      .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
    },
  };

  SDL_GPUVertexAttribute vadesc[] = {
    {
      .location = 0,
      .buffer_slot = 0,
      .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
      .offset = 0,
    },
  };

  SDL_GPUVertexInputState vinput = {
    .vertex_buffer_descriptions = vbdesc,
    .num_vertex_buffers = arraylen(vbdesc),
    .vertex_attributes = vadesc,
    .num_vertex_attributes = arraylen(vadesc)
  };

  SDL_GPURasterizerState raster = {
    .cull_mode = SDL_GPU_CULLMODE_NONE,
  };

  SDL_GPUColorTargetDescription color_desc[] = {
    {
      .format = state.sc_format,
      .blend_state = {
        .src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
        .dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
        .color_blend_op = SDL_GPU_BLENDOP_ADD,
        .src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
        .dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
        .alpha_blend_op = SDL_GPU_BLENDOP_ADD,
        .enable_blend = true,
      },
    },
  };

  SDL_GPUGraphicsPipelineTargetInfo target_info = {
    .color_target_descriptions = color_desc,
    .num_color_targets = arraylen(color_desc),
  };

  SDL_GPUGraphicsPipelineCreateInfo point_pipeline_info = {
    .vertex_shader = vshader,
    .fragment_shader = fshader,
    .vertex_input_state = vinput,
    .primitive_type = SDL_GPU_PRIMITIVETYPE_POINTLIST,
    .rasterizer_state = raster,
    .multisample_state = {.sample_count = state.sample_cnt},
    .target_info = target_info,
  };

  state.point.pipeline = SDL_CreateGPUGraphicsPipeline(state.dev,
      &point_pipeline_info);
  if (!state.point.pipeline)
    abort();

  SDL_ReleaseGPUShader(state.dev, vshader);
  SDL_ReleaseGPUShader(state.dev, fshader);
}

void init_test_pipeline(void)
{
  // init shaders
  size_t vsrc_sz; 
  void* vsrc = SDL_LoadFile("shaders/test.vert.spv", &vsrc_sz);

  SDL_GPUShaderCreateInfo vinfo = {
    .code_size = vsrc_sz,
    .code = (uint8_t *)vsrc,
    .entrypoint = "main",
    .format = SDL_GPU_SHADERFORMAT_SPIRV,
    .stage = SDL_GPU_SHADERSTAGE_VERTEX,
  };
  SDL_GPUShader *vshader = SDL_CreateGPUShader(state.dev, &vinfo);
  SDL_free(vsrc);

  if (!vshader)
    abort();

  size_t fsrc_sz; 
  void* fsrc = SDL_LoadFile("shaders/test.frag.spv", &fsrc_sz);

  SDL_GPUShaderCreateInfo finfo = {
    .code_size = fsrc_sz,
    .code = (uint8_t *)fsrc,
    .entrypoint = "main",
    .format = SDL_GPU_SHADERFORMAT_SPIRV,
    .stage = SDL_GPU_SHADERSTAGE_FRAGMENT,
  };
  SDL_GPUShader *fshader = SDL_CreateGPUShader(state.dev, &finfo);
  SDL_free(fsrc);

  if (!fshader)
    abort();

  SDL_GPUVertexInputState vinput = {};

  SDL_GPURasterizerState raster = {
    .cull_mode = SDL_GPU_CULLMODE_NONE,
  };

  SDL_GPUColorTargetDescription color_desc[] = {
    {
      .format = SDL_GetGPUSwapchainTextureFormat(state.dev, state.win),
      .blend_state = {
        .src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
        .dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
        .color_blend_op = SDL_GPU_BLENDOP_ADD,
        .src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
        .dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
        .alpha_blend_op = SDL_GPU_BLENDOP_ADD,
        .enable_blend = true,
      },
    },
  };

  SDL_GPUGraphicsPipelineTargetInfo target_info = {
    .color_target_descriptions = color_desc,
    .num_color_targets = arraylen(color_desc),
  };

  SDL_GPUGraphicsPipelineCreateInfo pipeline_info = {
    .vertex_shader = vshader,
    .fragment_shader = fshader,
    .vertex_input_state = vinput,
    .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
    .rasterizer_state = raster,
    .multisample_state = {.sample_count = state.sample_cnt},
    .target_info = target_info,
  };

  state.test_pipeline = SDL_CreateGPUGraphicsPipeline(state.dev,
      &pipeline_info);
  if (!state.test_pipeline)
    abort();

  SDL_ReleaseGPUShader(state.dev, vshader);
  SDL_ReleaseGPUShader(state.dev, fshader);
}

void init_win(void)
{
  float scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
  state.scale = scale;

  SDL_WindowFlags flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN
    | SDL_WINDOW_HIGH_PIXEL_DENSITY;

  int w = (int)(1024 * scale);
  int h = (int)(720 * scale);

  state.win = SDL_CreateWindow("esim", w, h, flags);
  if (!state.win)
    abort();

  SDL_ShowWindow(state.win);
}

void init_imgui(void)
{
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();

  ImGuiIO& io = ImGui::GetIO(); (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

  ImGui::StyleColorsDark();
  // ImGui::StyleColorsLight();

  ImGuiStyle& style = ImGui::GetStyle();
  style.ScaleAllSizes(state.scale);
  style.FontScaleDpi = state.scale;

  ImGui_ImplSDL3_InitForSDLGPU(state.win);
  ImGui_ImplSDLGPU3_InitInfo init_info = {};
  init_info.Device = state.dev;
  init_info.ColorTargetFormat =
    SDL_GetGPUSwapchainTextureFormat(state.dev, state.win);
  init_info.MSAASamples = state.sample_cnt;
  init_info.SwapchainComposition = SDL_GPU_SWAPCHAINCOMPOSITION_SDR;
  init_info.PresentMode = SDL_GPU_PRESENTMODE_VSYNC;
  ImGui_ImplSDLGPU3_Init(&init_info);
}

void init_render_txt(void)
{
  if (state.render_txt)
    SDL_ReleaseGPUTexture(state.dev, state.render_txt);
  if (state.resolve_txt)
    SDL_ReleaseGPUTexture(state.dev, state.resolve_txt);

  int w, h;
  SDL_GetWindowSizeInPixels(state.win, &w, &h);
  state.w = w;
  state.h = h;

  SDL_GPUTextureCreateInfo txt_info = {
    .type = SDL_GPU_TEXTURETYPE_2D,
    .format = state.sc_format,
    .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET,
    .width = (uint32_t)w,
    .height = (uint32_t)h,
    .layer_count_or_depth = 1,
    .num_levels = 1,
    .sample_count = state.sample_cnt
  };
  state.render_txt = SDL_CreateGPUTexture(state.dev, &txt_info);
  if (!state.render_txt)
    abort();

  txt_info = {
    .type = SDL_GPU_TEXTURETYPE_2D,
    .format = state.sc_format,
    .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER,
    .width = (uint32_t)w,
    .height = (uint32_t)h,
    .layer_count_or_depth = 1,
    .num_levels = 1,
  };

  state.resolve_txt = SDL_CreateGPUTexture(state.dev, &txt_info);
  if (!state.resolve_txt)
    abort();
}

void init_sdlgpu(void)
{
  state.dev = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV
      | SDL_GPU_SHADERFORMAT_DXIL | SDL_GPU_SHADERFORMAT_METALLIB,
      true, nullptr);
  if (!state.dev)
    abort();

  if (!SDL_ClaimWindowForGPUDevice(state.dev, state.win))
    abort();

  state.sample_cnt = SDL_GPU_SAMPLECOUNT_8;
  state.sc_format = SDL_GetGPUSwapchainTextureFormat(state.dev, state.win);
  SDL_SetGPUSwapchainParameters(state.dev, state.win,
      SDL_GPU_SWAPCHAINCOMPOSITION_SDR, SDL_GPU_PRESENTMODE_VSYNC);

  state.render_txt = NULL;
  state.resolve_txt = NULL;

  init_render_txt();
}

void point(vec2 pos)
{
  PointVertex *v = buffer_next(&state.point.vertices);
  v->pos = pos;
}

void vertex2d(vec2 pos, vec4 col)
{
  vec4 p = vec4_init(pos[0], pos[1], 0, 1);
  *buffer_next(&state.fill.vertices) = {
    .pos = p,
    .col = col,
  };
}

void circle(vec2 mid, float r, int sides, vec4 col)
{
  float dt = rad(360.f/sides);

  vec2 p0 = mid;
  p0[0] += r;

  vec2 pp = p0;

  for (int i = 1; i < sides; i++) {
    float s = sinf(i*dt);
    float c = cosf(i*dt);

    vec2 p = p0;

    p -= mid;
    p = vec2_init(p[0] * c - p[1] * s, p[0] * s + p[1] * c);
    p += mid;

    vertex2d(mid, col);
    vertex2d(pp, col);
    vertex2d(p, col);

    pp = p;
  }

  vertex2d(mid, col);
  vertex2d(pp, col);
  vertex2d(p0, col);
}

void triangle(vec2 beg, vec2 end, float w, vec4 col)
{
  vec2 D = normalize(end-beg);
  vec2 N = normalize(vec2_init(-D[1], D[0]));
  float h = length(end-beg);

  vertex2d(beg+D*h, col);
  vertex2d(beg+N*(w/2), col);
  vertex2d(beg-N*(w/2), col);
}

void triangle2(vec2 a, vec2 b, vec2 c, vec4 col)
{
  vertex2d(a, col);
  vertex2d(b, col);
  vertex2d(c, col);
}

void rect(vec2 beg, vec2 end, float w, vec4 col)
{
  vec2 D = normalize(end-beg);
  vec2 N = normalize(vec2_init(-D[1], D[0]));
  float h = length(end-beg);

  vertex2d((beg+D*h)+(N*w), col);
  vertex2d(beg+N*w, col);
  vertex2d(beg-N*w, col);
  vertex2d((end-D*h)-(N*w), col);
  vertex2d(end+N*w, col);
  vertex2d(end-N*w, col);
}

float line_point_distance(vec2 a, vec2 b, vec2 p)
{
  float n0 = fabsf((b[1]-a[1])*p[0] - (b[0]-a[0])*p[1] + b[0]*a[1] - b[1]*a[0]);
  float n1 = sqrtf(powf(b[1]-a[1], 2) + powf(b[0]-a[0], 2));

  return n0/n1;
}

mat4 cammat(void)
{
  mat4 A = translation(state.cam.pos);
  mat4 B = scaling(vec3_init(state.cam.scale, state.cam.scale, 1));
  mat4 C = translation(-state.cam.target);

  return mmul(mmul(A, B), C);
}

vec3 screen_to_world(vec3 p)
{
  mat4 C = cammat();
  mat4 M = invert(C);

  return mmulv3(M, p);
}

void render(void)
{
  ImGui::Render();
  ImDrawData *draw_data = ImGui::GetDrawData();
  const bool is_minimized =
    (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);

  SDL_GPUCommandBuffer *cb = SDL_AcquireGPUCommandBuffer(state.dev);

  Uint32 width, height;
  SDL_GPUTexture *swapchain_texture;
  SDL_WaitAndAcquireGPUSwapchainTexture(cb, state.win,
      &swapchain_texture, &width, &height);

  if (swapchain_texture && !is_minimized) {
    ImGui_ImplSDLGPU3_PrepareDrawData(draw_data, cb);

    // copy pass
    SDL_GPUCopyPass* cp = SDL_BeginGPUCopyPass(cb);

    if (state.fill.vertices.pos) {
      void* tbmem = SDL_MapGPUTransferBuffer(state.dev, state.fill.vtbuf, true);
      uint32_t tbsz = sizeof(FillVertex) * state.fill.vertices.pos;
      memcpy(tbmem, state.fill.vertices.data, tbsz);
      SDL_UnmapGPUTransferBuffer(state.dev, state.fill.vtbuf);

      SDL_GPUTransferBufferLocation location = {
        .transfer_buffer = state.fill.vtbuf,
      };

      SDL_GPUBufferRegion region = {
        .buffer = state.fill.vbuf,
        .size = tbsz,
      };

      SDL_UploadToGPUBuffer(cp, &location, &region, false);
    }

    if (state.point.vertices.pos) {
      void *tbmem = SDL_MapGPUTransferBuffer(state.dev, state.point.vtbuf, true);
      uint32_t tbsz = sizeof(PointVertex) * state.point.vertices.pos;
      memcpy(tbmem, state.point.vertices.data, tbsz);
      SDL_UnmapGPUTransferBuffer(state.dev, state.point.vtbuf);

      SDL_GPUTransferBufferLocation location = {
        .transfer_buffer = state.point.vtbuf,
      };

      SDL_GPUBufferRegion region = {
        .buffer = state.point.vbuf,
        .size = tbsz,
      };
      SDL_UploadToGPUBuffer(cp, &location, &region, false);
    }

    SDL_EndGPUCopyPass(cp);

    SDL_GPUColorTargetInfo target_info;
    SDL_GPURenderPass *rp;

    target_info = {
      .texture = state.render_txt,
      .clear_color = SDL_FColor{1.0,1.0,1.0,1.0},
      .load_op = SDL_GPU_LOADOP_CLEAR,
      .store_op = SDL_GPU_STOREOP_RESOLVE,
      .resolve_texture = state.resolve_txt,
    };
    rp = SDL_BeginGPURenderPass(cb, &target_info, 1, nullptr);

#if 0
      state.proj = orthographic_lhzo(-(width/2.f), (width/2.f),
            -(height/2.f), (height/2.f), 0.01, 100.0);
#else

      state.proj = orthographic_lhzo(0, width,
            height, 0, 0.01, 100.0);
#endif
    state.view = mmul(state.proj, cammat());
 
    // points
    if (1 && state.point.vertices.pos) {
      SDL_BindGPUGraphicsPipeline(rp, state.point.pipeline);

      SDL_GPUBufferBinding binding = {
        .buffer = state.point.vbuf,
        .offset = 0,
      };
      SDL_BindGPUVertexBuffers(rp, 0, &binding, 1);

      struct PointUbo {
        mat4 view;
        float pointsz;
      };

      PointUbo ub = {
        .view = state.view,
        .pointsz = state.grid.size*state.cam.scale,
      };
      SDL_PushGPUVertexUniformData(cb, 0, &ub, sizeof(ub));
      SDL_DrawGPUPrimitives(rp, state.point.vertices.pos, 1, 0, 0);
    }

    // fill
    if (state.fill.vertices.pos) {
      SDL_BindGPUGraphicsPipeline(rp, state.fill.pipeline);

      SDL_GPUBufferBinding binding = {
        .buffer = state.fill.vbuf,
        .offset = 0,
      };
      SDL_BindGPUVertexBuffers(rp, 0, &binding, 1);

      SDL_PushGPUVertexUniformData(cb, 0, &state.view, sizeof(state.view));

      SDL_DrawGPUPrimitives(rp, state.fill.vertices.pos, 1, 0, 0);
    }

    ImGui_ImplSDLGPU3_RenderDrawData(draw_data, cb, rp);

#if 0

    // test triangle
    SDL_BindGPUGraphicsPipeline(rp, state.test_pipeline);

    SDL_GPUViewport vp = {
      .x = wpos.x,
      .y = wpos.y,
      .w = wsiz.x,
      .h = wsiz.y,
      .min_depth = 0,
      .max_depth = 1,
    };
    SDL_SetGPUViewport(rp, &vp);

    SDL_DrawGPUPrimitives(rp, 3, 1, 0, 0);
#endif

   SDL_EndGPURenderPass(rp);

#if 1
		SDL_GPUBlitInfo blit_info;

		blit_info = {
      .source = {
        .texture = state.resolve_txt,
        .w = (uint32_t)state.w,
        .h = (uint32_t)state.h,
      },
      .destination = {
        .texture = swapchain_texture,
        .w = (uint32_t)state.w,
        .h = (uint32_t)state.h,
      },
      .load_op = SDL_GPU_LOADOP_DONT_CARE,
      .filter = SDL_GPU_FILTER_LINEAR
    };

		SDL_BlitGPUTexture(cb, &blit_info);
#endif
  }

  SDL_SubmitGPUCommandBuffer(cb);
}

void reset(void)
{
  buffer_reset(&state.fill.vertices);
  buffer_reset(&state.point.vertices);
  ImGui_ImplSDLGPU3_NewFrame();
  ImGui_ImplSDL3_NewFrame();
  ImGui::NewFrame();
}

vec2 qbezier(float t, vec2 p0, vec2 p1, vec2 p2)
{
  float u = 1.f-t;
  return u*u * p0 + 2 * u * t * p1 + (t*t) * p2;
}

void bezier(vec2 a, vec2 b, vec2 c, float w, int parts, vec4 col)
{
  float bd = 1.f/parts;
  vec2 p = qbezier(0, a, b, c);
  for (int i = 0; i < parts; i++) {
    vec2 cp = qbezier(i*bd+bd, a, b, c);
    rect(p, cp, w, col);
    p = cp;
  }
}

vec2 perpendicular(vec2 a, vec2 b)
{
  vec2 d = normalize(a - b);

  return normalize(vec2_init(-d[1], d[0]));
}

void bezier2(vec2 a, vec2 b, vec2 c, float w, int parts, vec4 col)
{
  float bd = 1.f/parts;

  for (int i = 0; i < parts; i++) {
    vec2 p0 = qbezier(i*bd, a, b, c);
    vec2 p1 = qbezier(i*bd+bd, a, b, c);

    vec2 n = perpendicular(p0, p1);

    if (i < parts-1) {
      vec2 t0 = qbezier(i*bd+bd, a, b, c);
      vec2 t1 = qbezier(i*bd+bd+bd, a, b, c);
      vec2 nn = perpendicular(t0, t1);
      n = (nn+n)/2;
    } else {
      vec2 t0 = qbezier(i*bd-bd, a, b, c);
      vec2 t1 = qbezier(i*bd+bd, a, b, c);
      vec2 nn = perpendicular(t0, t1);
      n = (nn+n)/2;
    }

    rect((p0+p1)/2, ((p0+p1)/2)-n*8, 0.5f, vec4_init(1,0,0,1));

    vertex2d(p0-n*w, col);
    vertex2d(p0+n*w, col);
    vertex2d(p1+n*w, col);

    vertex2d(p0-n*w, col);
    vertex2d(p1-n*w, col);
    vertex2d(p1+n*w, col);
  }
}

void bezier3(vec2 a, vec2 b, vec2 c, float w, vec4 col)
{
  const int parts = 16;
  float bd = 1.f/parts;

  vec2 normals[parts+1];
  for (int i = 1; i < parts; i++) {
    vec2 p0 = qbezier(i*bd-bd, a, b, c);
    vec2 p1 = qbezier(i*bd, a, b, c);
    vec2 p2 = qbezier(i*bd+bd, a, b, c);

    vec2 n0 = perpendicular(p0, p1);
    vec2 n1 = perpendicular(p1, p2);
    normals[i] = (n0+n1)/2;
  }

  {
    vec2 p0 = qbezier(0*bd, a, b, c);
    vec2 p1 = qbezier(1*bd, a, b, c);
    normals[0] = perpendicular(p0, p1);
  }
  {
    vec2 p0 = qbezier((parts-1)*bd, a, b, c);
    vec2 p1 = qbezier((parts)*bd, a, b, c);
    normals[parts] = perpendicular(p0, p1);
  }

  for (int i = 0; i < parts; i++) {
    vec2 p0 = qbezier(i*bd, a, b, c);
    vec2 p1 = qbezier(i*bd+bd, a, b, c);

    vec2 n0 = normals[i];
    vec2 n1 = normals[i+1];

    vertex2d(p0-n0*w, col);
    vertex2d(p0+n0*w, col);
    vertex2d(p1+n1*w, col);

    vertex2d(p0-n0*w, col);
    vertex2d(p1-n1*w, col);
    vertex2d(p1+n1*w, col);
  }
}


void start_new_shape(ShapeKind kind)
{
  memset(&state.new_shape, 0, sizeof(state.new_shape));
  state.new_shape_vertex_cnt = 0;
  state.new_shape.kind = kind;

  state.cursor.state = CursorState::PlaceShape;
}

void finish_new_shape(void)
{
  state.cursor.state = CursorState::Move;
  Shape *ns = buffer_next(&state.shapes);
  *ns = state.new_shape;
}

void add_new_shape_vertex(void)
{
  vec2 pos = state.cursor.pos;

  switch (state.new_shape.kind) {
  case ShapeKind::Circle:
    if (state.new_shape_vertex_cnt == 1) {
      state.new_shape.circle.r = length(state.new_shape.circle.pos - pos);
      finish_new_shape();
    } else {
      state.new_shape.circle.pos = pos;
    }
    break;
  case ShapeKind::Triangle:
    switch (state.new_shape_vertex_cnt) {
    case 0:
      state.new_shape.tri.a = pos;
      break;
    case 1:
      state.new_shape.tri.b = pos;
      break;
    case 2:
      state.new_shape.tri.c = pos;
      finish_new_shape();
      break;
    default:
      break;
    }
    break;
  case ShapeKind::Rectangle:
    switch (state.new_shape_vertex_cnt) {
    case 0:
      state.new_shape.rect.a = pos;
      break;
    case 1:
      state.new_shape.rect.b = pos;
      break;
    case 2:
      state.new_shape.rect.w = line_point_distance(state.new_shape.tri.a,
          state.new_shape.tri.b, state.cursor.pos);
      finish_new_shape();
      break;
    default:
      break;
    }
    break;
  case ShapeKind::Bezier:
    switch (state.new_shape_vertex_cnt) {
    case 0:
      state.new_shape.tri.a = pos;
      break;
    case 1:
      state.new_shape.tri.b = pos;
      break;
    case 2:
      state.new_shape.tri.c = pos;
      finish_new_shape();
      break;
    default:
      break;
    }
    break;
  default:
    break;
  }

  state.new_shape_vertex_cnt++;
}

void draw_shape_vertex(vec2 pos)
{
  circle(pos, 8.f/state.cam.scale, 16, vec4_init(1,0,0,0.5));
}

void draw_new_shape(void)
{
  if (state.cursor.state == CursorState::Move)
    return;

  vec4 newcol = vec4_init(0,0,1,0.5);
  switch (state.new_shape.kind) {
  case ShapeKind::Circle:
  {
    if (state.new_shape_vertex_cnt == 1) {
      vec2 p = state.new_shape.circle.pos;

      float r = length(p - state.cursor.pos);
      int sides = min(max(16, r/6), 32);
      circle(p, r, sides, newcol);

      if (ImGui::BeginTooltip()) {
        ImGui::Text("circle radius: %f", r);
        ImGui::EndTooltip();
      }
      draw_shape_vertex(p);
    }
    break;
  }
  case ShapeKind::Triangle:
    switch (state.new_shape_vertex_cnt) {
    case 1:
      draw_shape_vertex(state.new_shape.tri.a);
      break;
    case 2:
      draw_shape_vertex(state.new_shape.tri.a);
      draw_shape_vertex(state.new_shape.tri.b);
      triangle2(state.new_shape.tri.a, state.new_shape.tri.b,
          state.cursor.pos, newcol);
      break;
    default:
      break;
    }
    break;
  case ShapeKind::Rectangle:
    switch (state.new_shape_vertex_cnt) {
    case 1:
      draw_shape_vertex(state.new_shape.tri.a);
      break;
    case 2:
    {
      draw_shape_vertex(state.new_shape.tri.a);
      draw_shape_vertex(state.new_shape.tri.b);
      float w = line_point_distance(state.new_shape.tri.a,
          state.new_shape.tri.b, state.cursor.pos);

      if (ImGui::BeginTooltip()) {
        ImGui::Text("rectangle width: %f", w);
        ImGui::EndTooltip();
      }

      rect(state.new_shape.tri.a, state.new_shape.tri.b, w, newcol);
    }
      break;
    default:
      break;
    }
    break;
  case ShapeKind::Bezier:
    switch (state.new_shape_vertex_cnt) {
    case 1:
      draw_shape_vertex(state.new_shape.tri.a);
      break;
    case 2:
      draw_shape_vertex(state.new_shape.tri.a);
      draw_shape_vertex(state.new_shape.tri.b);
      bezier3(state.new_shape.tri.a, state.cursor.pos,
          state.new_shape.tri.b, 1.f, newcol);
      break;
    default:
      break;
    }
    break;
  default:
    break;
  }
}

void menu_bar(void)
{
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Create")) {}
      if (ImGui::MenuItem("Open", "Ctrl+O")) {}
      if (ImGui::MenuItem("Save", "Ctrl+S")) {}
      if (ImGui::MenuItem("Save as..")) {}
      ImGui::EndMenu();
    }

#if 0
    if (ImGui::BeginMenu("Components")) {
      if (ImGui::MenuItem("(w) Wire"))
        state.new_comp = ComponentKind::Wire;
      if (ImGui::MenuItem("(v) Voltage"))
        state.new_comp = ComponentKind::Voltage;
      if (ImGui::MenuItem("(i) Current"))
        state.new_comp = ComponentKind::Current;
      if (ImGui::MenuItem("(r) Resistor"))
        state.new_comp = ComponentKind::Resistor;
      if (ImGui::MenuItem("(c) Capacitor"))
        state.new_comp = ComponentKind::Capacitor;
      if (ImGui::MenuItem("(l) Inductor"))
        state.new_comp = ComponentKind::Inductor;
      if (ImGui::MenuItem("(d) Diode"))
        state.new_comp = ComponentKind::Diode;
      if (ImGui::MenuItem("(t) Transistor"))
        state.new_comp = ComponentKind::Transistor;
      ImGui::EndMenu();
    }
#endif

    if (ImGui::BeginMenu("Add")) {
      ImGui::SeparatorText("Models");
        if (ImGui::MenuItem("(r) Resistor"))
          printf("resistor\n");
      ImGui::SeparatorText("Shapes");
        if (ImGui::MenuItem("Circle"))
          start_new_shape(ShapeKind::Circle);
        if (ImGui::MenuItem("Triangle"))
          start_new_shape(ShapeKind::Triangle);
        if (ImGui::MenuItem("Rectangle"))
          start_new_shape(ShapeKind::Rectangle);
        if (ImGui::MenuItem("Bezier"))
          start_new_shape(ShapeKind::Bezier);
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Settings")) {
      state.en_settings_window = true;
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }
}

void settings_window(void)
{
  if (state.en_settings_window) {
    if (ImGui::Begin("Settings", &state.en_settings_window,
        ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::SliderInt("Grid width", &state.grid.width, 0, 5000, NULL);
      ImGui::SliderInt("Grid height", &state.grid.height, 0, 5000, NULL);
      ImGui::SliderFloat("Grid point size", &state.grid.size, 0.0f, 10.0f, NULL);
    }
    ImGui::End();
  }
}

void draw_grid(void)
{
  float w = state.grid.width;
  float h = state.grid.height;
  rect(vec2_init(0,0), vec2_init(w,0), 1.f, vec4_init(0,0,0,1));
  rect(vec2_init(0,0), vec2_init(0,h), 1.f, vec4_init(0,0,0,1));
  rect(vec2_init(w,0), vec2_init(w,h), 1.f, vec4_init(0,0,0,1));
  rect(vec2_init(0,h), vec2_init(w,h), 1.f, vec4_init(0,0,0,1));

  float tilesz = state.grid.tilesz;
  for (float y = 0; y < h; y+=tilesz)
    for (float x = 0; x < w; x+=tilesz)
      point(vec2_init(x, y));
}

void update_cursor(void)
{
  vec3 p = vec3_init(state.cursor.pos[0], state.cursor.pos[1], 0);
  vec3 v = screen_to_world(p);

  vec2 raw = vec2_init(v[0], v[1]);
  state.cursor.raw = raw;
  state.cursor.pos = raw;

  if (state.kb[SDL_SCANCODE_LCTRL]) {
    float x = roundf(v[0]/10)*10;
    float y = roundf(v[1]/10)*10;
    state.cursor.pos = vec2_init(x, y);
  }
  if (state.kb[SDL_SCANCODE_LSHIFT]) {
    float x = roundf(v[0]/1)*1;
    float y = roundf(v[1]/1)*1;
    state.cursor.pos = vec2_init(x, y);
  }

  float x = roundf(v[0]/10)*10;
  float y = roundf(v[1]/10)*10;
  state.cursor.grid = vec2_init(x, y)/10;
}

bool is_cursor_over_canvas(void)
{
  return !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);
}

void draw_cursor(void)
{
  if (!is_cursor_over_canvas())
    return;

  vec2 p = state.cursor.pos;

#if 1
  if (state.cursor.state == CursorState::Move) {
    circle(p, 3.f/state.cam.scale, 16, vec4_init(1,0,0,1));
    return;
  }
#endif

  float x = p[0];
  float y = p[1];

  float sz = 16.f/state.cam.scale;
  float w = 1.f/state.cam.scale;
  vec4 col = vec4_init(1.0,0,0,1.0);

  rect(vec2_init(x, y-sz), vec2_init(x, y+sz), w, col);
  rect(vec2_init(x-sz, y), vec2_init(x+sz, y), w, col);
}

void draw_shapes(void)
{
  vec4 col = vec4_init(0,0,0,1);

  for (size_t i = 0; i < state.shapes.pos; i++) {
    Shape *s = buffer_at(&state.shapes, i);
    switch (s->kind) {
    case ShapeKind::Circle:
      circle(s->circle.pos, s->circle.r, 24, col);
      circle(s->circle.pos, s->circle.r-2, 24, vec4_init(1,1,1,1));
      break;
    case ShapeKind::Triangle:
      triangle2(s->tri.a, s->tri.b, s->tri.c, col);
      break;
    case ShapeKind::Rectangle:
      rect(s->rect.a, s->rect.b, s->rect.w, col);
      break;
    case ShapeKind::Bezier:
      bezier3(s->tri.a, s->tri.c, s->tri.b, 1.f, col);
      break;
    default:
      break;
    }
  }
}

void draw_demo(void)
{
  const int N = 1000;
  float dt = 0.001;
  float data_x[N];
  float data_y[N];

  for (int i = 0; i < N; i++) {
    float x = (i+100)*dt;
    data_x[i] = x;
    data_y[i] = sinf(230.f*x)/x;
  }

  ImGui::Begin("Node 0 Voltage");
  if (ImPlot::BeginPlot("Voltage")) {
    ImPlot::PlotLine("", data_x, data_y, N);
    ImPlot::EndPlot();
  }
  ImGui::End();
}

void update(float dt)
{
  (void)dt;

  if (is_cursor_over_canvas())
    SDL_HideCursor();

  draw_grid();
  draw_shapes();
  draw_new_shape();

  menu_bar();
  settings_window();

  draw_cursor();

  draw_demo();

#if 0
  ImPlot::ShowDemoWindow();
  ImGui::ShowDemoWindow();
#endif
}

void on_left_click(void)
{
  switch (state.cursor.state) {
  case CursorState::Move:
    state.en_drag = true;
    break;
  case CursorState::PlaceShape:
    add_new_shape_vertex();
    break;
  case CursorState::PlaceModel:
    break;
  }
}

void on_left_unclick(void)
{
  switch (state.cursor.state) {
  case CursorState::Move:
    state.en_drag = false;
    break;
  case CursorState::PlaceShape:
    break;
  case CursorState::PlaceModel:
    break;
  }
}

void on_right_click(void)
{
  state.en_drag = true;
}

void on_right_unclick(void)
{
  state.en_drag = false;
}

int main(void)
{
  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD))
    abort();

  init_win();
  init_sdlgpu();
  init_imgui();
  init_fill();
  init_point();
  init_test_pipeline();

  state.start_time = SDL_GetTicks();
  state.prev_time = state.start_time;
  state.curr_time = state.start_time;
  state.grid.size = 1.f;
  state.grid.tilesz = 20.f;
  state.grid.width = 3000;
  state.grid.height = 2100;
  state.cam.scale = 1.f;
  buffer_init(&state.shapes);
  state.kb = SDL_GetKeyboardState(NULL);

  reset();

  bool done = false;
  while (!done) {
    SDL_Event evt;
    while (SDL_PollEvent(&evt)) {
      ImGui_ImplSDL3_ProcessEvent(&evt);
      switch (evt.type) {
      case SDL_EVENT_QUIT:
        done = true;
        break;
      case SDL_EVENT_WINDOW_RESIZED:
        init_render_txt();
        break;
      case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
        if (evt.window.windowID == SDL_GetWindowID(state.win))
          done = true;
        break;
      case SDL_EVENT_KEY_DOWN:
        if (is_cursor_over_canvas())
          if (evt.key.key == 'q')
            done = true;

        if (evt.key.key == SDLK_K)
          state.shapes.pos--;

        if (evt.key.key == SDLK_ESCAPE)
          state.cursor.state = CursorState::Move;

        break;
      case SDL_EVENT_KEY_UP:
        break;
      case SDL_EVENT_MOUSE_BUTTON_DOWN:
        if (evt.button.button == 3 && is_cursor_over_canvas())
          on_right_click();
        if (evt.button.button == 1 && is_cursor_over_canvas())
          on_left_click();
        break;
      case SDL_EVENT_MOUSE_BUTTON_UP:
        if (evt.button.button == 1)
          on_left_unclick();
        break;
      case SDL_EVENT_MOUSE_MOTION:
        state.cursor.pos[0] = evt.motion.x;
        state.cursor.pos[1] = evt.motion.y;
        if (state.en_drag) {
          state.cam.target[0] += (float)evt.motion.xrel * (-1.f/state.cam.scale);
          state.cam.target[1] += (float)evt.motion.yrel * (-1.f/state.cam.scale);
        }
        update_cursor();
        break;
      case SDL_EVENT_MOUSE_WHEEL:
      {
        if (is_cursor_over_canvas()) {
          state.cam.scale += state.cam.scale * (evt.wheel.y/10.f);

          vec3 p = vec3_init(evt.wheel.mouse_x, evt.wheel.mouse_y, 0);
          state.cam.target = screen_to_world(p);
          state.cam.pos = p;
        }

        break;
      }
      default:
        break;
      }
    }

    uint64_t now = SDL_GetTicks();
    state.curr_time = now;
    uint64_t idt = now - state.prev_time;
    float dt = (float)idt / 1000.f;
    update(dt);
    state.prev_time = now;
    state.elapsed_time = now - state.start_time;
    state.frame++;

    render();
    reset();
  }

  SDL_WaitForGPUIdle(state.dev);

  buffer_deinit(&state.shapes);

  ImGui_ImplSDL3_Shutdown();
  ImGui_ImplSDLGPU3_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();

  SDL_ReleaseGPUGraphicsPipeline(state.dev, state.test_pipeline);

  SDL_ReleaseGPUBuffer(state.dev, state.fill.vbuf);
  SDL_ReleaseGPUTransferBuffer(state.dev, state.fill.vtbuf);
  SDL_ReleaseGPUGraphicsPipeline(state.dev, state.fill.pipeline);
  buffer_deinit(&state.fill.vertices);

  SDL_ReleaseGPUBuffer(state.dev, state.point.vbuf);
  SDL_ReleaseGPUTransferBuffer(state.dev, state.point.vtbuf);
  SDL_ReleaseGPUGraphicsPipeline(state.dev, state.point.pipeline);
  buffer_deinit(&state.point.vertices);

  SDL_ReleaseGPUTexture(state.dev, state.render_txt);
  SDL_ReleaseGPUTexture(state.dev, state.resolve_txt);

  SDL_ReleaseWindowFromGPUDevice(state.dev, state.win);
  SDL_DestroyGPUDevice(state.dev);
  SDL_DestroyWindow(state.win);
  SDL_Quit();

  return 0;
}
