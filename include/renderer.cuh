#pragma once

#include "camera.cuh"
#include "scene.cuh"

#include <cuda_gl_interop.h>
#include <driver_types.h>
#include <vector>

// render configuration
struct RenderConfig {
        int image_w, image_h;   // final render resolution
        int samples_per_pixel;  // final render samples per pixel
        int max_depth;
};

class Renderer {
       public:
        RenderConfig config;
        GLuint gl_texture;
        int final_resolution_idx;

        // flags
        bool render_needs_update;
        bool scene_needs_update;

        // counter
        int sample_count;

        // video camera positions
        std::vector<Camera> camera_positions;

        Renderer();
        ~Renderer();

        // setup opengl texture and cuda buffers
        void init(const int width, const int height, const RenderConfig render_config);

        // render frame for real-time display
        void render_single_frame(const Scene &scene, const Camera &camera);

        // render full quality frame for exporting.
        void render_full_frame(const char file_path[], const Camera &camera);

        // render full quality video
        void render_video(const char name[]);

        void inline set_final_resolution(const int width, const int height) {
                config.image_w = width;
                config.image_h = height;
        }

       private:
        // window dimensions
        int window_w, window_h;

        cudaGraphicsResource *cuda_texture_resource;

        // cuda buffers
        float4 *accumulation_buffer;
        Scene *d_scene;
};
