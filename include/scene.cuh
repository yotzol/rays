#pragma once

#include "material.cuh"
#include "sphere.cuh"

// maximum number of objects and materials in the scene
const size_t MAX_SPHERES   = 1 << 16;
const size_t MAX_MATERIALS = 1 << 16;

class __align__(16) Scene {
       public:
        Sphere spheres[MAX_SPHERES];
        Material materials[MAX_MATERIALS];
        int num_spheres;
        int num_materials;

        __host__ __device__ Scene() : num_spheres(0), num_materials(0) {}

        // add a sphere to the scene
        __host__ void add_sphere(const Sphere &sphere);

        // add a material to the scene and return its index
        __host__ int add_material(const Material &material);

        // find the nearest intersection with any object in the scene
        __device__ bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const;
};
