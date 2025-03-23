#pragma once

#include "material.cuh"
#include "ray.cuh"

#include <cmath>

class __align__(16) Sphere {
       public:
        Ray center;
        float radius;
        int material_id;

        __host__ __device__ Sphere() : radius(0) {}

        // stationary
        __host__ __device__ Sphere(const Vec3 &static_center, float radius, const int material_id)
            : center(static_center, Vec3(0, 0, 0)), radius(std::fmax(radius, 0.0f)), material_id(material_id) {}
        // moving
        __host__ __device__ Sphere(const Vec3 &center1, const Vec3 &center2, float radius, const int material_id)
            : center(center1, center2 - center1), radius(std::fmax(radius, 0.0f)), material_id(material_id) {}

        __device__ bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const;
};
