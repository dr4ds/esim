OBJS+=ext/imgui.o ext/imgui_demo.o ext/imgui_draw.o ext/imgui_tables.o
OBJS+=ext/imgui_widgets.o ext/imgui_impl_sdl3.o ext/imgui_impl_sdlgpu3.o
OBJS+=ext/implot.o ext/implot_demo.o ext/implot_items.o
OBJS+=mathx.o

SHADERS+=shaders/fill.vert.spv shaders/fill.frag.spv
SHADERS+=shaders/point.vert.spv shaders/point.frag.spv
SHADERS+=shaders/test.vert.spv shaders/test.frag.spv

CXXFLAGS=-std=c++11 -I. -Iext -O3 -g -mfma -Wall -Wextra -Werror
LDFLAGS=-lSDL3 -lm
CXX=clang++
LD=clang++

%.o: ex/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

%.vert.spv: %.vert
	glslc -c $< -o $@

%.frag.spv: %.frag
	glslc -c $< -o $@

all: shaders sim

shaders: $(SHADERS)

sim: sim.o $(OBJS)
	$(LD) $(LDFLAGS) $^ -o $@

clean:
	rm -f *.o *.elf shaders/*.spv sim

.PHONY: all shaders clean
