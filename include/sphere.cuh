#ifndef SPHERE_CUH
#define SPHERE_CUH

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

        __device__ __forceinline__ bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const {
                Vec3 curr_center = center.at(ray.time);

                Vec3 oc      = ray.origin - curr_center;
                float a      = ray.direction.length_squared();
                float half_b = dot(oc, ray.direction);
                float c      = oc.length_squared() - radius * radius;

                float discriminant = half_b * half_b - a * c;
                if (discriminant < 0) return false;

                float sqrtd = sqrtf(discriminant);
                float root  = (-half_b - sqrtd) / a;
                if (root < t_min || root > t_max) {
                        root = (-half_b + sqrtd) / a;
                        if (root < t_min || root > t_max) return false;
                }

                rec.t               = root;
                rec.point           = ray.at(rec.t);
                Vec3 outward_normal = (rec.point - curr_center) / radius;
                rec.set_face_normal(ray, outward_normal);
                rec.material_id = material_id;

                return true;
        }
};

#endif  // SPHERE_CUH
