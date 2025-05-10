#pragma once

#include "utils.hpp"

#include <assert.h>
#include <cmath>

// Vec3 structure. 3 floats + 1 pad.
struct __align__(16) Vec3 {
        float x, y, z, pad;

        // Constructors.
        __host__ __device__ Vec3() : x(0), y(0), z(0), pad(0) {}
        __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z), pad(0) {}
        __host__ __device__ Vec3(float f) : x(f), y(f), z(f), pad(0) {}

        // Mutable indexing.
        CUDA_INLINE float &operator[](int index) {
                assert(0 <= index && index <= 2);
                switch (index) {
                        case 0 : return x;
                        case 1 : return y;
                        case 2 : return z;
                        default: return x;  // Unreachable.
                }
        }

        // Immutable indexing.
        CUDA_INLINE const float &operator[](int index) const {
                assert(0 <= index && index <= 2);
                switch (index) {
                        case 0 : return x;
                        case 1 : return y;
                        case 2 : return z;
                        default: return x;  // Unreachable.
                }
        }

        // Negate.
        CUDA_INLINE const Vec3 operator-() const {
                return Vec3(-x, -y, -z);
        }

        // Add two vectors.
        CUDA_INLINE Vec3 operator+(const Vec3 &v) const {
                return Vec3(x + v.x, y + v.y, z + v.z);
        }

        // Add vectors in-place.
        CUDA_INLINE void operator+=(const Vec3 &v) {
                x += v.x;
                y += v.y;
                z += v.z;
        }

        // Subtract two vectors.
        CUDA_INLINE Vec3 operator-(const Vec3 &v) const {
                return Vec3(x - v.x, y - v.y, z - v.z);
        }

        // Subctract vectors in-place.
        CUDA_INLINE void operator-=(const Vec3 &v) {
                x -= v.x;
                y -= v.y;
                z -= v.z;
        }

        // Multiply two vectors.
        CUDA_INLINE Vec3 operator*(const Vec3 &v) const {
                return Vec3(x * v.x, y * v.y, z * v.z);
        }

        // Multiply vectors in-place.
        CUDA_INLINE void operator*=(const Vec3 &v) {
                x *= v.x;
                y *= v.y;
                z *= v.z;
        }

        // Multiply vector with scalar
        CUDA_INLINE Vec3 operator*(const float t) const {
                return Vec3(x * t, y * t, z * t);
        }

        // Scale in-place.
        CUDA_INLINE void operator*=(const float t) {
                x *= t;
                y *= t;
                z *= t;
        }

        // Divide vector with scalar.
        CUDA_INLINE Vec3 operator/(const float t) const {
                return Vec3(x / t, y / t, z / t);
        }

        // Scale in-place.
        CUDA_INLINE void operator/=(const float t) {
                x /= t;
                y /= t;
                z /= t;
        }

        CUDA_INLINE float length_squared() const {
                return x * x + y * y + z * z;
        }

        CUDA_INLINE float length() const {
                return sqrtf(length_squared());
        }
};

// Multiply scalar with vector.
CUDA_INLINE Vec3 operator*(const float t, const Vec3 v) {
        return Vec3(v.x * t, v.y * t, v.z * t);
}

// Dot product.
CUDA_INLINE float dot(const Vec3 &a, const Vec3 &b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Cross product.
CUDA_INLINE Vec3 cross(const Vec3 &a, const Vec3 &b) {
        return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

CUDA_INLINE Vec3 normalize(const Vec3 &v) {
        float len = v.length();
        return v / len;
}

// Linear Interpolation.
CUDA_INLINE Vec3 lerp(const Vec3 &a, const Vec3 &b, const float t) {
        return a + (b - a) * t;
}

// ============================================================================
//  Ray structure
// ============================================================================
struct __align__(16) Ray {
        Vec3 orig;
        Vec3 dir;

        __host__ __device__ Ray() {}

        __host__ __device__ Ray(const Vec3 &origin, const Vec3 &direction) : orig(origin), dir(direction) {}

        __device__ __forceinline__ Vec3 at(float t) const {
                return orig + dir * t;
        }
};

// ============================================================================
//  Utilities
// ============================================================================

#ifdef __CUDACC__

// Generate a random vector in the unit sphere.
__device__ __forceinline__ Vec3 random_in_unit_sphere(curandState *state) {
        while (true) {
                Vec3 p(randf(-1, 1, state), randf(-1, 1, state), randf(-1, 1, state));
                if (p.length_squared() < 1) return p;
        }
}

// Generate a random vector in the unit disk.
__device__ __forceinline__ Vec3 random_in_unit_disk(curandState *state) {
        while (true) {
                Vec3 p(randf(-1, 1, state), randf(-1, 1, state), 0);
                if (p.length_squared() < 1) return p;
        }
}

// Generate a random unit vector.
__device__ __forceinline__ Vec3 random_unit_vector(curandState *state) {
        return normalize(random_in_unit_sphere(state));
}

// Generate a reflected vector.
__device__ __forceinline__ Vec3 reflect(const Vec3 &v, const Vec3 &n) {
        return v - n * 2 * dot(v, n);
}

// Generate a refracted vector.
__device__ __forceinline__ Vec3 refract(const Vec3 &uv, const Vec3 &n, float etai_over_etat) {
        float cos_theta     = fminf(dot(uv * -1.0f, n), 1.0f);
        Vec3 r_out_perp     = (uv + n * cos_theta) * etai_over_etat;
        Vec3 r_out_parallel = n * -sqrtf(fabsf(1.0f - r_out_perp.length_squared()));
        return r_out_perp + r_out_parallel;
}

#endif
