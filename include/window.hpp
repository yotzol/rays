#pragma once

#include "camera.hpp"
#include "renderer.hpp"
#include "scene.hpp"

#include <GLFW/glfw3.h>
#include <atomic>

namespace window {

extern GLFWwindow *window;

extern std::atomic<bool> is_exporting;

void create_window(int width, int height);
void main_loop(Scene &scene, Camera &camera, Renderer &renderer);

}  // namespace window
