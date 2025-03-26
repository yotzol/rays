#include "scene.cuh"

#include <assert.h>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

__host__ void Scene::add_sphere(const Sphere &sphere) {
        if (num_spheres < MAX_SPHERES) {
                spheres[num_spheres++] = sphere;
        }
        assert(num_spheres < MAX_SPHERES);
}

__host__ int Scene::add_material(const Material &material) {
        if (num_materials < MAX_MATERIALS) {
                materials[num_materials] = material;
                return num_materials++;
        }
        assert(num_materials < MAX_MATERIALS);
        return -1;
}

__device__ bool Scene::hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const {
        int stack[32];  // fixed-size stack
        int stack_ptr        = 0;
        stack[stack_ptr++]   = root_idx;
        bool hit_anything    = false;
        float closest_so_far = t_max;
        HitRecord temp_rec;

        while (stack_ptr > 0) {
                int node_idx        = stack[--stack_ptr];
                const BvhNode &node = bvh_nodes[node_idx];

                if (node.bbox.hit(ray, t_min, t_max, rec)) {
                        if (node.is_leaf) {
                                for (int i = 0; i < node.leaf.count; i++) {
                                        int sphere_idx = node.leaf.idx_start + i;
                                        if (spheres[sphere_idx].hit(ray, t_min, closest_so_far, temp_rec)) {
                                                hit_anything   = true;
                                                closest_so_far = temp_rec.t;
                                                rec            = temp_rec;
                                        }
                                }
                        } else {
                                // internal: push children
                                stack[stack_ptr++] = node.inner.idx_l;
                                stack[stack_ptr++] = node.inner.idx_r;
                        }
                }
        }
        return hit_anything;
}
