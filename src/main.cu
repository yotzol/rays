#include "camera.cuh"
#include "default_scenes.cuh"
#include "renderer.cuh"
#include "scene.cuh"
#include "utils.cuh"
#include "vec3.cuh"
#include "window.cuh"

#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int WINDOW_W = 1280;
const int WINDOW_H =  720;

int main() {
        // seed the random number generator
        srand(time(NULL));

        // render settings
        int image_width  = 1080;
        int image_height = 720;

        RenderConfig render_config;
        render_config.image_w           = image_width;
        render_config.image_h           = image_height;
        render_config.samples_per_pixel = 128;
        render_config.max_depth         = 128;

        // camera settings
        Camera camera(Vec3(13, 2, 3),                            // lookfrom
                      Vec3(0, 0, 0),                             // lookat
                      Vec3(0, 1, 0),                             // up vector
                      20.0f,                                     // vertical fov
                      float(image_width) / float(image_height),  // aspect ratio
                      0.1f,                                      // aperture
                      10.0f                                      // focus distance
        );

        Scene scene;
        default_scenes::env_map_test(scene);
        scene.build_bvh();

        create_window(WINDOW_W, WINDOW_H);

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
        renderer.init(WINDOW_W, WINDOW_H, render_config);

        main_loop(scene, camera, renderer);

        glfwTerminate();

        return 0;
}
