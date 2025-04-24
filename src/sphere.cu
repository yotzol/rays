#include "object.cuh"

__host__ Object sphere_new(const Vec3 static_center, float radius, const int material_id) {
        Object obj;

        obj.type            = SPHERE;
        obj.material_id     = material_id;
        obj.sphere.center   = Ray(static_center, Vec3(0.0f, 0.0f, 0.0f));
        obj.sphere.radius   = std::fmax(radius, 0.0f);
        obj.rotation_center = static_center;

        return obj;
}

__host__ Object sphere_moving(const Vec3 &center1, const Vec3 &center2, float radius, const int material_id) {
        Object obj;

        obj.type            = SPHERE;
        obj.material_id     = material_id;
        obj.sphere.center   = Ray(center1, center2 - center1);
        obj.sphere.radius   = std::fmax(radius, 0.0f);
        obj.rotation_center = center1;

        return obj;
}

__device__ bool Object::sphere_hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const {
        Vec3 curr_center = sphere.center.at(ray.time);

        Vec3 oc      = ray.origin - curr_center;
        float a      = ray.direction.length_squared();
        float half_b = dot(oc, ray.direction);
        float c      = oc.length_squared() - sphere.radius * sphere.radius;

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
        Vec3 outward_normal = (rec.point - curr_center) / sphere.radius;
        rec.set_face_normal(ray, outward_normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.material_id = material_id;

        return true;
}
