#pragma once

#include "camera.cuh"
#include "renderer.cuh"

namespace gui {

extern bool button_export_clicked;

void setup_imgui();

void render_imgui(Renderer &renderer, Camera &camera);

}  // namespace gui
