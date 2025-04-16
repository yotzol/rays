#pragma once

#include "camera.cuh"
#include "renderer.cuh"

namespace gui {

void setup_imgui();

void render_imgui(Renderer &renderer, Camera &camera);

}  // namespace gui
