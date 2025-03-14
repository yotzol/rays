#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "ray.cuh"
#include "utils.cuh"

struct __align__(16) HitRecord {
        Vec3 point;
        Vec3 normal;
        float t;
        bool front_face;
        int material_id;

        __device__ __forceinline__ void set_face_normal(const Ray &ray, const Vec3 &outward_normal) {
                front_face = dot(ray.direction, outward_normal) < 0;
                normal     = front_face ? outward_normal : outward_normal * -1.0f;
        }
};

enum MaterialType { LAMBERTIAN, METAL, DIELECTRIC };

class __align__(16) Material {
       public:
        MaterialType type;
        Vec3 albedo;    // for Lambertian and Metal
        float fuzz;     // for Metal (0 = no fuzz, 1 = max fuzz)
        float ref_idx;  // for Dielectric

        __host__ __device__ Material() : type(LAMBERTIAN), fuzz(0), ref_idx(1) {}

        __host__ __device__ Material(MaterialType t, const Vec3 &a, float f = 0, float ri = 1)
            : type(t), albedo(a), fuzz(f), ref_idx(ri) {}

        // generic scatter function that dispatches to the appropriate material type
        __device__ bool scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered,
                                RandState *rand_state) const {
                switch (type) {
                        case LAMBERTIAN: return scatter_lambertian(ray_in, rec, attenuation, scattered, rand_state);
                        case METAL     : return scatter_metal(ray_in, rec, attenuation, scattered, rand_state);
                        case DIELECTRIC: return scatter_dielectric(ray_in, rec, attenuation, scattered, rand_state);
                        default        : return false;
                }
        }

       private:
        __device__ bool scatter_lambertian(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered,
                                           RandState *rand_state) const {
                Vec3 scatter_direction = rec.normal + random_unit_vector(rand_state);

                // catch degenerate scatter direction
                if (scatter_direction.length_squared() < 0.001f) scatter_direction = rec.normal;

                scattered   = Ray(rec.point, scatter_direction, ray_in.time);
                attenuation = albedo;
                return true;
        }

        __device__ bool scatter_metal(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered,
                                      RandState *rand_state) const {
                Vec3 reflected = reflect(normalize(ray_in.direction), rec.normal);
                scattered      = Ray(rec.point, reflected + random_in_unit_sphere(rand_state) * fuzz, ray_in.time);
                attenuation    = albedo;
                return (dot(scattered.direction, rec.normal) > 0);
        }

        __device__ bool scatter_dielectric(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered,
                                           RandState *rand_state) const {
                attenuation            = Vec3(1.0f, 1.0f, 1.0f);  // glass doesn't absorb light
                float refraction_ratio = rec.front_face ? (1.0f / ref_idx) : ref_idx;

                Vec3 unit_direction = normalize(ray_in.direction);
                float cos_theta     = fminf(dot(unit_direction * -1.0f, rec.normal), 1.0f);
                float sin_theta     = sqrtf(1.0f - cos_theta * cos_theta);

                bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
                Vec3 direction;

                if (cannot_refract || schlick(cos_theta, refraction_ratio) > random_float(rand_state))
                        direction = reflect(unit_direction, rec.normal);
                else
                        direction = refract(unit_direction, rec.normal, refraction_ratio);

                scattered = Ray(rec.point, direction, ray_in.time);
                return true;
        }
};

#endif  // MATERIAL_CUH
