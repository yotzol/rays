#include "camera.hpp"
#include "default_scenes.hpp"
#include "renderer.hpp"
#include "scene.hpp"
#include "utils.hpp"
#include "window.hpp"

#include <cstdio>
#include <stdio.h>
#include <time.h>

const int WINDOW_W = 1280;
const int WINDOW_H = 720;

int main() {
        // Seed the random number generator.
        srand((unsigned int)time(NULL));

        // Render settings.
        int preview_w = 600;
        int preview_h = 600;

        RenderConfig render_config;
        render_config.window_w          = preview_w;
        render_config.window_h          = preview_h;
        render_config.samples_per_pixel = 128;
        render_config.max_depth         = 128;

        Camera camera;
        camera.aspect_ratio = (float)preview_w / (float)preview_h;

        Scene scene;
        default_scenes::book2_final_scene(scene, camera);
        scene.build_bvh();

        window::create_window(WINDOW_W, WINDOW_H);

        CHECK_CUDA_ERROR(cudaSetDevice(0));

        // Force CUDA context creation (cudaFree(0) is a no-op that initializes the context).
        CHECK_CUDA_ERROR(cudaFree(0));

        // Query CUDA devices compatible with opengl.
        unsigned int gl_device_count;
        CHECK_CUDA_ERROR(cudaGLGetDevices(&gl_device_count, nullptr, 0, cudaGLDeviceListAll));
        if (gl_device_count == 0) {
                fprintf(stderr, "No CUDA devices support OpenGL interop\n");
                glfwTerminate();
                exit(EXIT_FAILURE);
        }

        printf("Found %u CUDA device(s) supporting OpenGL interop\n", gl_device_count);

        Renderer renderer = Renderer();
        renderer.init(render_config);

        window::main_loop(scene, camera, renderer);

        glfwTerminate();

        return 0;
}
