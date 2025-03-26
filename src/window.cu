#include "window.cuh"
#include "camera.cuh"
#include "renderer.cuh"
#include "scene.cuh"
#include "gui.cuh"

#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <stdio.h>

GLFWwindow *window = nullptr;

void create_window(const int width, const int height) {
        if (!glfwInit()) {
                fprintf(stderr, "Failed to initialize GLFW\n");
                exit(EXIT_FAILURE);
        }

        window = glfwCreateWindow(width, height, "Rays", nullptr, nullptr);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        if (!window) {
                fprintf(stderr, "Failed to create GLFW window\n");
                glfwTerminate();
                exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(window);

        gui::setup_imgui();
}

static bool handle_camera_movement(Camera &camera);

void main_loop(Scene &scene, Camera &camera, Renderer &renderer) {
        bool camera_moved = false;

        while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();

                if (!renderer.is_rendering) {
                        // exit program on esc
                        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;

                        if (handle_camera_movement(camera)) renderer.render_needs_update = true;

                        renderer.render_single_frame(scene, camera);
                } else {
                        renderer.render_full_frame("output.png", scene, camera);
                        renderer.is_rendering = false;
                }

                gui::render_imgui(renderer, camera);

                glfwSwapBuffers(window);
        }

        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

static bool handle_camera_movement(Camera &camera) {
        bool camera_moved = false;

        // camera position
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { camera.move(CameraMovement::FORWARD);  camera_moved = true; }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { camera.move(CameraMovement::LEFT);     camera_moved = true; }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { camera.move(CameraMovement::BACKWARD); camera_moved = true; }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { camera.move(CameraMovement::RIGHT);    camera_moved = true; }

        if (glfwGetKey(window, GLFW_KEY_SPACE)      == GLFW_PRESS) { camera.move(CameraMovement::UP);    camera_moved = true; }
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) { camera.move(CameraMovement::DOWN);  camera_moved = true; }

        // tilt
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) { camera.tilt(false); camera_moved = true; }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) { camera.tilt(true);  camera_moved = true; }

        // camera angle
        static float last_mouse_x           = 0.0f;
        static float last_mouse_y           = 0.0f;
        static bool first_mouse             = true;
        static bool is_mouse_control_active = false;

        double mouse_x, mouse_y;
        glfwGetCursorPos(window, &mouse_x, &mouse_y);
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
                if (!is_mouse_control_active) {
                        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);  // hide cursor
                        last_mouse_x            = mouse_x;
                        last_mouse_y            = mouse_y;
                        first_mouse             = true;
                        is_mouse_control_active = true;
                }

                // calculate mouse movement delta
                if (first_mouse) {
                        last_mouse_x = mouse_x;
                        last_mouse_y = mouse_y;
                        first_mouse  = false;
                }

                float delta_x = float(mouse_x - last_mouse_x) * camera.mouse_sensitivity;
                float delta_y = float(last_mouse_y - mouse_y) * camera.mouse_sensitivity;  // y coords are reversed

                if (delta_x != 0.0f && delta_y != 0.0f) {
                        camera.process_mouse_movement(delta_x, delta_y);
                        last_mouse_x = mouse_x;
                        last_mouse_y = mouse_y;
                        camera_moved = true;
                }
        } else if (is_mouse_control_active) {
                // release mouse control when right button is released
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                is_mouse_control_active = false;
        }

        return camera_moved;
}
