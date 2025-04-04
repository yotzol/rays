#pragma once

#include "vec3.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <stb_image.h>

enum TextureType { SOLID, CHECKER, IMAGE };

struct __align__(16) Texture {
        TextureType type;
        Vec3 color;                   // solid, checker
        Vec3 color2;                  // checker
        float scale;                  // checker
        cudaTextureObject_t tex_obj;  // image

        __host__ Texture() : type(SOLID), color(1.0f), color2(0), scale(0), tex_obj(0) {}

        __device__ Vec3 value(float u, float v, const Vec3 p) const;
};

__host__ Texture texture_solid(const Vec3 &c);

__host__ Texture texture_checker(const Vec3 &c, const Vec3 &c2, float s);

__host__ Texture texture_image(const char path[]);
