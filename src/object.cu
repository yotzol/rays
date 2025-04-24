#include "object.cuh"

__device__ bool Object::hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const {
        bool hit = false;

        // translation
        Vec3 offset_origin = ray.origin - translation;

        Ray offset_ray(offset_origin, ray.direction, ray.time);

        // world -> object
        if (rotation != 0) {
                Vec3 origin_rel = offset_origin - rotation_center;

                // inverse rotation
                Vec3 origin_local = Vec3(cos_theta * origin_rel.x - sin_theta * origin_rel.z, origin_rel.y,
                                         sin_theta * origin_rel.x + cos_theta * origin_rel.z) +
                                    rotation_center;
                Vec3 direction_local = Vec3(cos_theta * ray.direction.x - sin_theta * ray.direction.z, ray.direction.y,
                                            sin_theta * ray.direction.x + cos_theta * ray.direction.z);

                offset_ray = Ray(origin_local, direction_local, ray.time);
        }

        switch (type) {
                case SPHERE: hit = sphere_hit(offset_ray, t_min, t_max, rec); break;
                case QUAD  : hit = quad_hit  (offset_ray, t_min, t_max, rec); break;
        }

        if (!hit) return false;

        // object -> world
        if (rotation != 0) {
                Vec3 point_rel = rec.point - rotation_center;

                // forward rotation
                rec.point = Vec3(cos_theta * point_rel.x + sin_theta * point_rel.z, point_rel.y,
                                -sin_theta * point_rel.x + cos_theta * point_rel.z);
                rec.point += rotation_center;

                rec.normal = Vec3(cos_theta * rec.normal.x + sin_theta * rec.normal.z, rec.normal.y,
                                 -sin_theta * rec.normal.x + cos_theta * rec.normal.z);
        }
        rec.point += translation;

        return true;
}
