#pragma once

#include "object.cuh"
#include "vec3.cuh"

#include <vector>

struct Aabb {
        Vec3 min, max;

        __host__ Aabb() { pad(); }
        __host__ Aabb(Vec3 min, Vec3 max) : min(min), max(max) { pad(); }
        __host__ Aabb(const Object *spheres, const std::vector<int> &indices, int start, int end);

        __device__ bool hit(const Ray &ray, float t_min, float t_max) const;

        __host__ static Aabb from_object(const Object &sphere);
        __host__ static Aabb merge(const Aabb &a, const Aabb &b);

        __host__ __forceinline__ Aabb operator+(const Vec3 &offset) {
                return Aabb(min + offset, max + offset);
        }

        __host__ __forceinline__ void operator+=(const Vec3 &offset) {
                min += offset;
                max += offset;
        }

       private:
        __host__ void pad();
};
