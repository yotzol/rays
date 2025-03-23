#pragma once

#include "camera.cuh"
#include "scene.cuh"

#include <cuda_gl_interop.h>
#include <driver_types.h>

// render configuration
struct RenderConfig {
        int image_w, image_h;   // final render resolution
        int samples_per_pixel;  // final render samples per pixel
        int max_depth;
};

class Renderer {
       public:
        RenderConfig config;

        Renderer();
        ~Renderer();

        // setup opengl texture and cuda buffers
        void init(const int width, const int height, const RenderConfig render_config);

        void render(const Scene &scene, const Camera &camera, bool reset_accumulation);

        GLuint get_texture() const {
                return gl_texture;
        }

       private:
        // window dimensions
        int window_w, window_h;

        // opengl display
        GLuint gl_texture;
        cudaGraphicsResource *cuda_texture_resource;

        // cuda buffers
        float4 *accumulation_buffer;
        unsigned char *display_buffer;  // mapped opengl texture
        Scene *d_scene;

        // counter
        int sample_count;

        // flags
        bool scene_needs_update;
};
