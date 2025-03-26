#pragma once

#include "vec3.cuh"

#include <curand_kernel.h>
#include <stdio.h>

typedef curandState RandState;

// cuda error checking macro
#define CHECK_CUDA_ERROR(call)                                                                                         \
        {                                                                                                              \
                cudaError_t err = call;                                                                                \
                if (err != cudaSuccess) {                                                                              \
                        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__,                       \
                                cudaGetErrorString(err));                                                              \
                        cudaDeviceReset();                                                                             \
                        exit(EXIT_FAILURE);                                                                            \
                }                                                                                                      \
        }

const float FMAX = std::numeric_limits<float>::max();
const float PI   = 3.141592653589f;

__host__ __device__ __forceinline__ float degrees(float radians) {
        return radians * 180.0f / PI;
}

__host__ __device__ __forceinline__ float radians(float degrees) {
        return degrees * PI / 180.0f;
}

__device__ __forceinline__ void random_init(RandState *state, int seed, int index) {
        curand_init(seed, index, 0, state);
}

// random float in [0,1)
__device__ __forceinline__ float random_float(RandState *state) {
        return curand_uniform(state);
}

__host__ __forceinline__ float random_float() {
        return (float)rand() / RAND_MAX;
}

// random float in [min,max)
__device__ __forceinline__ float random_float(float min, float max, RandState *state) {
        return min + (max - min) * random_float(state);
}

__host__ __forceinline__ float random_float(float min, float max) {
        return min + (max - min) * random_float();
}

__device__ __forceinline__ Vec3 random_in_unit_sphere(RandState *state) {
        while (true) {
                Vec3 p(random_float(-1, 1, state), random_float(-1, 1, state), random_float(-1, 1, state));
                if (p.length_squared() < 1) return p;
        }
}

// generate a random vector in a unit disk (for depth of field)
__device__ __forceinline__ Vec3 random_in_unit_disk(RandState *state) {
        while (true) {
                Vec3 p(random_float(-1, 1, state), random_float(-1, 1, state), 0);
                if (p.length_squared() < 1) return p;
        }
}

// generate a random unit vector (for lambertian scattering)
__device__ __forceinline__ Vec3 random_unit_vector(RandState *state) {
        return normalize(random_in_unit_sphere(state));
}

// generate a reflected vector (for dielectric)
__device__ __forceinline__ Vec3 reflect(const Vec3 &v, const Vec3 &n) {
        return v - n * 2 * dot(v, n);
}

// refraction calculation (for dielectric)
__device__ __forceinline__ Vec3 refract(const Vec3 &uv, const Vec3 &n, float etai_over_etat) {
        float cos_theta     = fminf(dot(uv * -1.0f, n), 1.0f);
        Vec3 r_out_perp     = (uv + n * cos_theta) * etai_over_etat;
        Vec3 r_out_parallel = n * -sqrtf(fabsf(1.0f - r_out_perp.length_squared()));
        return r_out_perp + r_out_parallel;
}

// schlick approximation for reflectance
__device__ __forceinline__ float schlick(float cosine, float ref_idx) {
        float r0 = (1 - ref_idx) / (1 + ref_idx);
        r0       = r0 * r0;
        return r0 + (1 - r0) * powf((1 - cosine), 5);
}

// clamp a value between min and max
__device__ __forceinline__ float clamp(float x, float min, float max) {
        return x < min ? min : (x > max ? max : x);
}

// convert float rgb color (0-1) to 32-bit rgba
__device__ __forceinline__ unsigned int make_color(float r, float g, float b) {
        r = clamp(r, 0.0f, 1.0f);
        g = clamp(g, 0.0f, 1.0f);
        b = clamp(b, 0.0f, 1.0f);

        unsigned int ir = static_cast<unsigned int>(255.99f * r);
        unsigned int ig = static_cast<unsigned int>(255.99f * g);
        unsigned int ib = static_cast<unsigned int>(255.99f * b);

        return 0xFF000000 | (ib << 16) | (ig << 8) | ir;
}
