#pragma once

#include "camera.hpp"
#include "scene.hpp"

#include <cuda_gl_interop.h>
#include <vector>

// Render configuration.
struct RenderConfig {
        int image_w, image_h;    // Final render dimensions.
        int window_w, window_h;  // Window preview dimensions.
        int samples_per_pixel;   // Final render samples per pixel.
        int max_depth;
};

class Renderer {
       public:
        RenderConfig config;
        GLuint gl_texture;
        int final_resolution_idx;

        // Flags.
        bool render_needs_update;
        bool scene_needs_update;

        int sample_count;  // Counter.

        // Camera video keyframes.
        std::vector<Camera> camera_positions;

        Renderer();
        ~Renderer();

        // Setup OpenGL texture and CUDA buffers.
        void init(const RenderConfig render_config);

        // Render a single pass into the accumulation buffer for real-time visualization.
        void render_single_frame(const Scene &scene, const Camera &camera);

        // Render full quality frame for exporting.
        void render_full_frame(const char file_path[], const Camera &camera);

        // Render full quality video.
        void render_video(const char name[]);

        void inline set_final_resolution(const int width, const int height) {
                config.image_w = width;
                config.image_h = height;
        }

       private:
        cudaGraphicsResource *cuda_texture_resource;

        float4 *accumulation_buffer;  // CUDA buffer.
        Scene *d_scene;               // Device scene.
};
