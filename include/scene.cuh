#pragma once

#include "bvh.cuh"
#include "material.cuh"
#include "object.cuh"

#include <vector>

// maximum number of objects and materials in the scene
const size_t MAX_OBJECTS   = 1 << 12;
const size_t MAX_MATERIALS = 1 << 12;
const size_t MAX_NODES     = (MAX_OBJECTS * 2) ;

struct __align__(16) Scene {
        Object objects[MAX_OBJECTS];
        Material materials[MAX_MATERIALS];
        BvhNode bvh_nodes[MAX_NODES];
        int num_objects;
        int num_materials;
        int root_idx;

        cudaTextureObject_t env_map;
        int env_w, env_h, env_channels;

        bool no_environment;
        Vec3 background_color;

        __host__ Scene() : num_objects(0), num_materials(0), root_idx(0), env_map(0), env_w(0), env_h(0), env_channels(0), no_environment(false), background_color(0.0f, 0.0f, 0.0f) {}

        // add an object to the scene
        __host__ void add_object(Object obj, float rotation = 0, Vec3 translation = Vec3());
        __host__ void add_object(Object obj, float rotation, Vec3 translation, Vec3 rotation_center);

        // add a material to the scene and return its index
        __host__ int add_material(const Material &material);

        // create box with given material
        __host__ void add_box(const Vec3 a, const Vec3 b, const int material_id, float rotation = 0, Vec3 translation = Vec3());

        // create bvh from objects
        __host__ void build_bvh();
        __host__ int build_bvh_recursive(std::vector<BvhNode> &nodes, std::vector<int> &indices, int start, int end);

        // load environment map
        __host__ void set_env_map(const char path[]);

        // find the nearest intersection with any object in the scene
        __device__ bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const;
};
