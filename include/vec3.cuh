#pragma once

#include <assert.h>

#include <cmath>
#include <cuda_runtime.h>

struct __align__(16) Vec3 {
        float x, y, z;
        float pad;

        // constructors
        __host__ __device__ Vec3() : x(0), y(0), z(0), pad(0) {}
        __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z), pad(0) {}
        __host__ __device__ Vec3(float f) : x(f), y(f), z(f), pad(0) {}

        // operator overloading
        __host__ __device__ __forceinline__ const float &operator[](int index) const {
                switch (index) {
                        case 0 : return x;
                        case 1 : return y;
                        case 2 : return z;
                        default: assert(0 <= index && index <= 2); return x;
                }
        }

        __host__ __device__ __forceinline__ Vec3 operator-() {
                return Vec3(-x, -y, -z);
        }

        __host__ __device__ __forceinline__ Vec3 operator+(const Vec3 &v) const {
                return Vec3(x + v.x, y + v.y, z + v.z);
        }

        __host__ __device__ __forceinline__ void operator+=(const Vec3 &v) {
                x += v.x;
                y += v.y;
                z += v.z;
        }

        __host__ __device__ __forceinline__ Vec3 operator-(const Vec3 &v) const {
                return Vec3(x - v.x, y - v.y, z - v.z);
        }

        __host__ __device__ __forceinline__ void operator-=(const Vec3 &v) {
                x -= v.x;
                y -= v.y;
                z -= v.z;
        }

        __host__ __device__ __forceinline__ Vec3 operator*(const float t) const {
                return Vec3(x * t, y * t, z * t);
        }

        __host__ __device__ __forceinline__ void operator*=(const float t) {
                x *= t;
                y *= t;
                z *= t;
        }

        __host__ __device__ __forceinline__ Vec3 operator*(const Vec3 v) const {
                return Vec3(x * v.x, y * v.y, z * v.z);
        }

        __host__ __device__ __forceinline__ void operator*=(const Vec3 &v) {
                x *= v.x;
                y *= v.y;
                z *= v.z;
        }

        __host__ __device__ __forceinline__ Vec3 operator/(const float t) const {
                float inv_t = 1.0f / t;
                return Vec3(x * inv_t, y * inv_t, z * inv_t);
        }

        __host__ __device__ __forceinline__ void operator/=(const float t) {
                x /= t;
                y /= t;
                z /= t;
        }

        // length
        __host__ __device__ __forceinline__ float length_squared() const {
                return x * x + y * y + z * z;
        }

        __host__ __device__ __forceinline__ float length() const {
                return sqrtf(length_squared());
        }
};

// non-member functions
__host__ __device__ __forceinline__ float dot(const Vec3 &a, const Vec3 &b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ __forceinline__ Vec3 normalize(const Vec3 &v) {
        float len = v.length();
        return v / len;
}

__host__ __device__ __forceinline__ Vec3 cross(const Vec3 &a, const Vec3 &b) {
        return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
