#include "object.hpp"

__device__ bool Object::hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const {
        bool hit = false;

        // Translation.
        Vec3 offset_origin = ray.orig - translation;

        Ray offset_ray(offset_origin, ray.dir);

        // World -> object.
        if (rotation != 0) {
                Vec3 origin_rel = offset_origin - rotation_center;

                // Inverse rotation.
                Vec3 origin_local = Vec3(cos_theta * origin_rel.x - sin_theta * origin_rel.z, origin_rel.y,
                                         sin_theta * origin_rel.x + cos_theta * origin_rel.z) +
                                    rotation_center;
                Vec3 direction_local = Vec3(cos_theta * ray.dir.x - sin_theta * ray.dir.z, ray.dir.y,
                                            sin_theta * ray.dir.x + cos_theta * ray.dir.z);

                offset_ray = Ray(origin_local, direction_local);
        }

        switch (type) {
                case SPHERE: hit = sphere_hit(offset_ray, t_min, t_max, rec); break;
                case QUAD  : hit = quad_hit(offset_ray, t_min, t_max, rec); break;
        }

        if (!hit) return false;

        // Object -> world.
        if (rotation != 0) {
                Vec3 point_rel = rec.point - rotation_center;

                // Forward rotation.
                rec.point = Vec3(cos_theta * point_rel.x + sin_theta * point_rel.z, point_rel.y,
                                 -sin_theta * point_rel.x + cos_theta * point_rel.z);
                rec.point += rotation_center;

                rec.normal = Vec3(cos_theta * rec.normal.x + sin_theta * rec.normal.z, rec.normal.y,
                                  -sin_theta * rec.normal.x + cos_theta * rec.normal.z);
        }
        rec.point += translation;

        return true;
}

__device__ bool Object::sphere_hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const {
        Vec3 curr_center = sphere.center.at(0.0f);

        Vec3 oc      = ray.orig - curr_center;
        float a      = ray.dir.length_squared();
        float half_b = dot(oc, ray.dir);
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

__device__ bool Object::quad_hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const {
        float denom = dot(quad.normal, ray.dir);

        if (std::fabs(denom) < 1e-8) return false;  // Ray is parallel to plane.

        float t = (quad.d - dot(quad.normal, ray.orig)) / denom;
        if (t < t_min || t > t_max) return false;  // Hit outside ray interval.

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
