#pragma once

#include "bvh.cuh"
#include "material.cuh"
#include "sphere.cuh"

#include <vector>

// maximum number of objects and materials in the scene
const size_t MAX_SPHERES   = 1 << 10;
const size_t MAX_MATERIALS = 1 << 10;
const size_t MAX_NODES     = (MAX_SPHERES * 2) ;

struct __align__(16) Scene {
        Sphere spheres[MAX_SPHERES];
        Material materials[MAX_MATERIALS];
        BvhNode bvh_nodes[MAX_NODES];
        int num_spheres;
        int num_materials;
        int root_idx;

        cudaTextureObject_t env_map;
        int env_w, env_h, env_channels;

        __host__ Scene() : num_spheres(0), num_materials(0), root_idx(0), env_map(0), env_w(0), env_h(0), env_channels(0) {}

        // add a sphere to the scene
        __host__ void add_sphere(const Sphere &sphere);

        // add a material to the scene and return its index
        __host__ int add_material(const Material &material);

        // create bvh from objects
        __host__ void build_bvh();
        __host__ int build_bvh_recursive(std::vector<BvhNode> &nodes, std::vector<int> &indices, int start, int end);

        // load environment map
        __host__ void set_env_map(const char path[]);

        // find the nearest intersection with any object in the scene
        __device__ bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const;
};
