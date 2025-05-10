#pragma once

#include "camera.hpp"
#include "renderer.hpp"

namespace gui {

void setup_imgui();

void render_imgui(Renderer &renderer, Camera &camera);

}  // namespace gui
