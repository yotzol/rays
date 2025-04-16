#pragma once

#include "ray.cuh"
#include "texture.cuh"
#include "utils.cuh"

struct __align__(16) HitRecord {
        Vec3 point;
        Vec3 normal;
        float t;
        bool front_face;
        int material_id;
        float u, v;

        __device__ __forceinline__ void set_face_normal(const Ray &ray, const Vec3 &outward_normal) {
                front_face = dot(ray.direction, outward_normal) < 0;
                normal     = front_face ? outward_normal : outward_normal * -1.0f;
        }
};

enum MaterialType { LAMBERTIAN, METAL, DIELECTRIC, DIFFUSE_LIGHT };

class __align__(16) Material {
       public:
        MaterialType type;
        union {
                struct {
                        Texture texture;
                } lambertian;
                struct {
                        Vec3 albedo;
                        float fuzz;  // 0 = no fuzz, 1 = max fuzz
                } metal;
                struct {
                        float ref_idx;
                } dielectric;
                struct {
                        Texture texture;
                } diffuse_light;
        };

        __host__ Material() {}; 

        // generic scatter function that dispatches to the appropriate material type
        __device__ bool scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered,
                                RandState *rand_state) const {
                switch (type) {
                        case LAMBERTIAN   : return scatter_lambertian(ray_in, rec, attenuation, scattered, rand_state);
                        case METAL        : return scatter_metal     (ray_in, rec, attenuation, scattered, rand_state);
                        case DIELECTRIC   : return scatter_dielectric(ray_in, rec, attenuation, scattered, rand_state);
                        case DIFFUSE_LIGHT: return false;
                };
                return false;
        }

        // generic emitted function that dispatches to the appropriate material type
        __device__ Vec3 emitted(float u, float v, const Vec3 p) const {
                switch (type) {
                        case LAMBERTIAN   : return false;
                        case METAL        : return false;
                        case DIELECTRIC   : return false;
                        case DIFFUSE_LIGHT: return emitted_diffuse_light(u, v, p);
                };
                return false;
        }

       private:
        __device__ __forceinline__ bool scatter_lambertian(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered,
                                           RandState *rand_state) const {
                Vec3 scatter_direction = rec.normal + random_unit_vector(rand_state);

                // catch degenerate scatter direction
                if (scatter_direction.length_squared() < 0.001f) scatter_direction = rec.normal;

                scattered   = Ray(rec.point, scatter_direction, ray_in.time);
                attenuation = lambertian.texture.value(rec.u, rec.v, rec.point);
                return true;
        }

        __device__ __forceinline__ bool scatter_metal(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered,
                                      RandState *rand_state) const {
                Vec3 reflected = reflect(normalize(ray_in.direction), rec.normal);
                scattered      = Ray(rec.point, reflected + random_in_unit_sphere(rand_state) * metal.fuzz, ray_in.time);
                attenuation    = metal.albedo;
                return (dot(scattered.direction, rec.normal) > 0);
        }

        __device__ __forceinline__ bool scatter_dielectric(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered,
                                           RandState *rand_state) const {
                attenuation            = Vec3(1.0f, 1.0f, 1.0f);  // glass doesn't absorb light
                float refraction_ratio = rec.front_face ? (1.0f / dielectric.ref_idx) : dielectric.ref_idx;

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

        __device__ __forceinline__ Vec3 emitted_diffuse_light(float u, float v, const Vec3 p) const {
                return diffuse_light.texture.value(u, v, p);
        }
};

Material lambertian_new(Texture texture);
Material metal_new(Vec3 albedo, float fuzz);
Material dielectric_new(float refraction_index);
Material diffuse_light_new(Texture texture);
