#pragma once

#include <cuda_runtime.h>

// Load file from assets/textures/ directory into CUDA texture.
cudaTextureObject_t load_texture(const char filename[]);

// Load file from assets/envmaps/ directory into CUDA texture.
cudaTextureObject_t load_envmap(const char filename[]);

// Load file from path into CUDA texture.
cudaTextureObject_t load_asset(const char path[]);

// Save a framebuffer to a PNG file.
__host__ bool save_png(const char *filename, unsigned int *framebuffer, int width, int height);

#ifdef __CUDACC__

#include "utils.hpp"

// Convert float rgb color (0-1) to 32-bit RGBA.
__device__ __forceinline__ unsigned int make_color(float r, float g, float b) {
        r = clamp(r, 0.0f, 1.0f);
        g = clamp(g, 0.0f, 1.0f);
        b = clamp(b, 0.0f, 1.0f);

        const unsigned int ir = static_cast<unsigned int>(255.99f * r);
        const unsigned int ig = static_cast<unsigned int>(255.99f * g);
        const unsigned int ib = static_cast<unsigned int>(255.99f * b);

        return 0xFF000000 | (ib << 16) | (ig << 8) | ir;
}

#endif
