#include "window.hpp"

#include "camera.hpp"
#include "gui.hpp"
#include "renderer.hpp"
#include "scene.hpp"

#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <atomic>
#include <cuda_gl_interop.h>
#include <stdio.h>

namespace window {

GLFWwindow *window = nullptr;

std::atomic<bool> is_exporting(false);

void create_window(const int width, const int height) {
        if (!glfwInit()) {
                fprintf(stderr, "Failed to initialize GLFW\n");
                exit(EXIT_FAILURE);
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(width, height, "Rays", nullptr, nullptr);

        if (!window) {
                fprintf(stderr, "Failed to create GLFW window\n");
                glfwTerminate();
                exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(window);

        gui::setup_imgui();
}

void handle_camera_movement(Camera &camera);

void main_loop(Scene &scene, Camera &camera, Renderer &renderer) {
        while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();

                // Real-time render (if not exporting).
                if (!is_exporting.load()) {
                        // Exit program on ESC.
                        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;

                        // Reset renderer if camera moved.
                        handle_camera_movement(camera);
                        if (camera.moved) {
                                renderer.render_needs_update = true;
                                camera.moved                 = false;
                        }

                        // Accumulate.
                        renderer.render_single_frame(scene, camera);
                }

                gui::render_imgui(renderer, camera);

                glfwSwapBuffers(window);
        }

        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

void handle_camera_movement(Camera &camera) {
        static float last_mouse_x           = 0.0f;
        static float last_mouse_y           = 0.0f;
        static bool first_mouse             = true;
        static bool is_mouse_control_active = false;

        if (is_mouse_control_active) {
                // Camera position.
                if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera.move(CameraMovement::FORWARD);
                if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera.move(CameraMovement::LEFT);
                if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera.move(CameraMovement::BACKWARD);
                if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera.move(CameraMovement::RIGHT);

                if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) camera.move(CameraMovement::UP);
                if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) camera.move(CameraMovement::DOWN);

                // Tilt.
                if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) camera.tilt(false);
        }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) camera.tilt(true);

        double mouse_x, mouse_y;
        glfwGetCursorPos(window, &mouse_x, &mouse_y);
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
                if (!is_mouse_control_active) {
                        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);  // Hide cursor.
                        last_mouse_x            = (float)mouse_x;
                        last_mouse_y            = (float)mouse_y;
                        first_mouse             = true;
                        is_mouse_control_active = true;
                }

                // Mouse movement delta.
                if (first_mouse) {
                        last_mouse_x = (float)mouse_x;
                        last_mouse_y = (float)mouse_y;
                        first_mouse  = false;
                }
                float delta_x = float(mouse_x - last_mouse_x) * camera.mouse_sensitivity;
                float delta_y = float(last_mouse_y - mouse_y) * camera.mouse_sensitivity;  // Y coords are reversed.

                if (delta_x != 0.0f && delta_y != 0.0f) {
                        camera.process_mouse_movement(delta_x, delta_y);
                        last_mouse_x = (float)mouse_x;
                        last_mouse_y = (float)mouse_y;
                }
        } else if (is_mouse_control_active) {
                // Release mouse control when right button is released.
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                is_mouse_control_active = false;
        }
}

}  // namespace window
