#pragma once

#include "image.hpp"
#include "vec3.hpp"

#include <cmath>
#include <cuda_runtime.h>

enum TextureType { SOLID, CHECKER, IMAGE };

struct __align__(16) Texture {
        TextureType type;
        union {
                struct {
                        Vec3 color;
                } solid;
                struct {
                        Vec3 color1;
                        Vec3 color2;
                        float scale;
                } checker;
                struct {
                        cudaTextureObject_t tex_obj;
                } image;
        };

        __host__ Texture() : type(SOLID), solid({1.0f}) {}

// CUDA guard required because of the built-in tex2D function.
#ifdef __CUDACC__
        __device__ __forceinline__ Vec3 value(float u, float v, const Vec3 p) const {
                switch (type) {
                        case SOLID  : return solid.color;
                        case CHECKER: {
                                float sines = sinf(checker.scale * p.x) * sinf(checker.scale * p.y) *
                                              sinf(checker.scale * p.z);
                                return sines < 0 ? checker.color1 : checker.color2;
                        }
                        case IMAGE: {
                                float4 c = tex2D<float4>(image.tex_obj, u, v);
                                return Vec3(c.x, c.y, c.z);
                        }
                        default: return Vec3(0, 0, 0);
                }
        }
#endif

};

// Create new solid texture from color.
__host__ inline Texture texture_solid(const Vec3 &albedo) {
        Texture texture;
        texture.type        = SOLID;
        texture.solid.color = albedo;
        return texture;
}

// Create new checker texture from two colors.
__host__ inline Texture texture_checker(const Vec3 &c, const Vec3 &c2, float s) {
        Texture t;
        t.type           = CHECKER;
        t.checker.color1 = c;
        t.checker.color2 = c2;
        t.checker.scale  = s;
        return t;
}

// Create new image texture from image path.
__host__ inline Texture texture_image(const char filename[]) {
        Texture texture;
        texture.type          = IMAGE;
        texture.image.tex_obj = load_texture(filename);
        return texture;
}
