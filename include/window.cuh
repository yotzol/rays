#pragma once

#include "camera.cuh"
#include "renderer.cuh"
#include "scene.cuh"

#include <GLFW/glfw3.h>

extern GLFWwindow *window;

void create_window(int width, int height);
void main_loop(Scene &scene, Camera &camera, Renderer &renderer);
