#pragma once

#include "sphere.cuh"
#include "vec3.cuh"

#include <vector>

struct Aabb {
        Vec3 min, max;

        __host__ Aabb() {}
        __host__ Aabb(Vec3 min, Vec3 max) : min(min), max(max) {}
        __host__ Aabb(const Sphere *spheres, const std::vector<int> &indices, int start, int end);

        __device__ bool hit(const Ray &ray, float t_min, float t_max) const;

        __host__ static Aabb from_sphere(const Sphere &sphere);
        __host__ static Aabb merge(const Aabb &a, const Aabb &b);
};
