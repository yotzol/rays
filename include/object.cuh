#pragma once

#include "material.cuh"
#include "ray.cuh"
#include "utils.cuh"

enum ObjectType { SPHERE, QUAD };

struct __align__(16) Object {
        ObjectType type;

        int material_id;
        union {
                struct {
                        Ray center;
                        float radius;
                } sphere;
                struct {
                        Vec3 q, u, v, w;
                        Vec3 normal;
                        float d;
                } quad;
        };

        __host__ Object() {}

        __device__ bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const;
        __device__ bool sphere_hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const;
        __device__ bool quad_hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const;

        __host__ Vec3 center();
};

__host__ Object sphere_new(const Vec3 static_center, float radius, const int material_id);
__host__ Object sphere_moving(const Vec3 &center1, const Vec3 &center2, float radius, const int material_id);
__host__ Object quad_new(const Vec3 q, const Vec3 u, const Vec3 v, const int material_id);

__device__ __forceinline__ void get_sphere_uv(const Vec3 p, float &u, float &v) {
        const float theta = acos(-p.y);
        const float phi   = atan2(-p.z, p.x) + PI;

        u = clamp(phi / TAU, 0, 1);
        v = 1 - clamp(theta / PI, 0, 1);
}
