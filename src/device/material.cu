#include "material.hpp"

// Generic scatter function that dispatches to the appropriate material type.
__device__ bool Material::scatter(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation, Ray &scattered,
                                  curandState *rand_state) const {
        switch (type) {
                case LAMBERTIAN   : return scatter_lambertian(ray_in, rec, attenuation, scattered, rand_state);
                case METAL        : return scatter_metal(ray_in, rec, attenuation, scattered, rand_state);
                case DIELECTRIC   : return scatter_dielectric(ray_in, rec, attenuation, scattered, rand_state);
                case DIFFUSE_LIGHT: return false;
        };
        return false;
}

// Generic emitted function that dispatches to the appropriate material type.
__device__ Vec3 Material::emitted(float u, float v, const Vec3 p) const {
        switch (type) {
                case LAMBERTIAN   : return false;
                case METAL        : return false;
                case DIELECTRIC   : return false;
                case DIFFUSE_LIGHT: return emitted_diffuse_light(u, v, p);
        };
        return false;
}
__device__ __forceinline__ bool Material::scatter_lambertian(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation,
                                                             Ray &scattered, curandState *rand_state) const {
        Vec3 scatter_direction = rec.normal + random_unit_vector(rand_state);

        // Catch degenerate scatter direction.
        if (scatter_direction.length_squared() < 0.001f) scatter_direction = rec.normal;

        scattered   = Ray(rec.point, scatter_direction);
        attenuation = lambertian.texture.value(rec.u, rec.v, rec.point);
        return true;
}

__device__ __forceinline__ bool Material::scatter_metal(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation,
                                                        Ray &scattered, curandState *rand_state) const {
        Vec3 reflected = reflect(normalize(ray_in.dir), rec.normal);
        scattered      = Ray(rec.point, reflected + random_in_unit_sphere(rand_state) * metal.fuzz);
        attenuation    = metal.albedo;
        return (dot(scattered.dir, rec.normal) > 0);
}

// Schlick approximation for reflectance.
__device__ __forceinline__ float schlick(float cosine, float ref_idx) {
        float r0 = (1 - ref_idx) / (1 + ref_idx);
        r0       = r0 * r0;
        return r0 + (1 - r0) * powf((1 - cosine), 5);
}

__device__ __forceinline__ bool Material::scatter_dielectric(const Ray &ray_in, const HitRecord &rec, Vec3 &attenuation,
                                                             Ray &scattered, curandState *rand_state) const {
        attenuation            = Vec3(1.0f, 1.0f, 1.0f);  // Glass doesn't absorb light.
        float refraction_ratio = rec.front_face ? (1.0f / dielectric.ref_idx) : dielectric.ref_idx;

        Vec3 unit_direction = normalize(ray_in.dir);
        float cos_theta     = fminf(dot(unit_direction * -1.0f, rec.normal), 1.0f);
        float sin_theta     = sqrtf(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        Vec3 direction;

        if (cannot_refract || schlick(cos_theta, refraction_ratio) > randf(rand_state))
                direction = reflect(unit_direction, rec.normal);
        else
                direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = Ray(rec.point, direction);
        return true;
}

__device__ __forceinline__ Vec3 Material::emitted_diffuse_light(float u, float v, const Vec3 p) const {
        return diffuse_light.texture.value(u, v, p);
}
