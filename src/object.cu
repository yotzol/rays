#include "object.cuh"

__device__ bool Object::hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const {
        switch (type) {
                case SPHERE: return sphere_hit(ray, t_min, t_max, rec);
                case QUAD  : return quad_hit(ray, t_min, t_max, rec);
        }
        return false;
}

__host__ Vec3 Object::center() {
        switch (type) {
                case SPHERE: return sphere.center.origin;
                case QUAD  : return quad.q + quad.u / 2 + quad.v / 2;
        };
        return Vec3();
}
