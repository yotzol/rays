#include "camera.cuh"
#include "default_scenes.cuh"
#include "renderer.cuh"
#include "scene.cuh"
#include "utils.cuh"
#include "vec3.cuh"
#include "window.cuh"

#include <cstdio>
#include <stdio.h>
#include <time.h>

const int WINDOW_W = 1280;
const int WINDOW_H =  720;

int main() {
        // seed the random number generator
        srand(time(NULL));

        // render settings
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
        default_scenes::cornell_box(scene, camera);
        scene.build_bvh();

        window::create_window(WINDOW_W, WINDOW_H);

        CHECK_CUDA_ERROR(cudaSetDevice(0));

        // force cuda context creation (cudafree(0) is a no-op that initializes the context)
        CHECK_CUDA_ERROR(cudaFree(0));

        // query cuda devices compatible with opengl
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
