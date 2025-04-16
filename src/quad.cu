#include "object.cuh"
#include <cmath>
#include <cstdio>

__host__ Object quad_new(const Vec3 q, const Vec3 u, const Vec3 v, const int material_id) {
        Object obj;
        obj.type = QUAD;
        obj.material_id = material_id;
        obj.quad.q = q;
        obj.quad.u = u;
        obj.quad.v = v;

        Vec3 n          = cross(u, v);
        obj.quad.normal = normalize(n);
        obj.quad.d      = dot(obj.quad.normal, q);
        obj.quad.w      = n / dot(n, n);

        return obj;
}

__device__ bool Object::quad_hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const {
        float denom = dot(quad.normal, ray.direction);

        if (std::fabs(denom) < 1e-8) return false;  // parallel to plane

        float t = (quad.d - dot(quad.normal, ray.origin)) / denom;
        if (t < t_min || t > t_max) return false;  // outside ray interval

        Vec3 intersection = ray.at(t);

        Vec3 planar_hit_p = intersection - quad.q;

        float a = dot(quad.w, cross(planar_hit_p, quad.v));
        float b = dot(quad.w, cross(quad.u, planar_hit_p));

        if ((a < 0 || a > 1) || (b < 0 || b > 1)) return false;

        rec.u = a;
        rec.v = b;
        rec.t = t;

        rec.point       = intersection;
        rec.material_id = material_id;
        rec.set_face_normal(ray, quad.normal);

        return true;
}
