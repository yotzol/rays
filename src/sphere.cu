#include "sphere.cuh"

__device__ bool Sphere::hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const {
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
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.material_id = material_id;

        return true;
}

__device__ __forceinline__ void Sphere::get_sphere_uv(const Vec3 p, float &u, float &v) const {
        const float theta = acos(-p.y);
        const float phi   = atan2(-p.z, p.x) + PI;

        u = clamp(phi / TAU, 0, 1);
        v = 1 - clamp(theta / PI, 0, 1);
}
