#ifndef SCENE_CUH
#define SCENE_CUH

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
        __host__ void add_sphere(const Sphere &sphere) {
                if (num_spheres < MAX_SPHERES) {
                        spheres[num_spheres++] = sphere;
                }
        }

        // add a material to the scene and return its index
        __host__ int add_material(const Material &material) {
                if (num_materials < MAX_MATERIALS) {
                        materials[num_materials] = material;
                        return num_materials++;
                }
                return -1;  // too many materials
        }

        // find the nearest intersection with any object in the scene
        __device__ bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const {
                HitRecord temp_rec;
                bool hit_anything    = false;
                float closest_so_far = t_max;

                for (int i = 0; i < num_spheres; i++) {
                        if (spheres[i].hit(ray, t_min, closest_so_far, temp_rec)) {
                                hit_anything   = true;
                                closest_so_far = temp_rec.t;
                                rec            = temp_rec;
                        }
                }

                return hit_anything;
        }
};

#endif  // SCENE_CUH
