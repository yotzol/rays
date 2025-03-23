#include "scene.cuh"

#include <assert.h>

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
