#pragma once

#include "texture.hpp"

struct __align__(16) HitRecord {
        Vec3 point;
        Vec3 normal;
        float t;
        bool front_face;
        int material_id;
        float u, v;

        __device__ __forceinline__ void set_face_normal(const Ray &ray, const Vec3 &outward_normal) {
                front_face = dot(ray.dir, outward_normal) < 0;
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
                        float fuzz;  // 0 = no fuzz, 1 = max fuzz.
                } metal;
                struct {
                        float ref_idx;
                } dielectric;
                struct {
                        Texture texture;
                } diffuse_light;
        };

        __host__ Material() {};

        // Generic scatter function that dispatches to the appropriate material type.
        __device__ bool scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered,
                                curandState *rand_state) const;

        // Generic emitted function that dispatches to the appropriate material type.
        __device__ Vec3 emitted(float u, float v, const Vec3 p) const;

       private:
        __device__ __forceinline__ bool scatter_lambertian(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation,
                                                           Ray &scattered, curandState *rand_state) const;

        __device__ bool scatter_metal(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered,
                                      curandState *rand_state) const;

        __device__ bool scatter_dielectric(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered,
                                           curandState *rand_state) const;

        __device__ Vec3 emitted_diffuse_light(float u, float v, const Vec3 p) const;
};

inline Material lambertian_new(Texture texture) {
        Material m           = Material();
        m.type               = LAMBERTIAN;
        m.lambertian.texture = texture;
        return m;
}

inline Material metal_new(Vec3 albedo, float fuzz) {
        Material m     = Material();
        m.type         = METAL;
        m.metal.albedo = albedo;
        m.metal.fuzz   = fuzz;
        return m;
}

inline Material dielectric_new(float refraction_index) {
        Material m           = Material();
        m.type               = DIELECTRIC;
        m.dielectric.ref_idx = refraction_index;
        return m;
}

inline Material diffuse_light_new(Texture texture) {
        Material m              = Material();
        m.type                  = DIFFUSE_LIGHT;
        m.diffuse_light.texture = texture;
        return m;
}
