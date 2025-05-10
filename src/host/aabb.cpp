#include "aabb.hpp"

__host__ Aabb::Aabb(const Object *objects, const std::vector<int> &idxs, int start, int end) {
        if (start >= end) {
                min = Vec3(0.0f, 0.0f, 0.0f);
                max = Vec3(0.0f, 0.0f, 0.f);
                return;
        }

        Aabb box = Aabb::from_object(objects[idxs[(size_t)start]]);
        for (int i = start + 1; i < end; ++i) {
                Aabb obj_box = Aabb::from_object(objects[idxs[(size_t)i]]);
                box          = Aabb::merge(box, obj_box);
        }

        min = box.min;
        max = box.max;
}

__host__ Aabb Aabb::merge(const Aabb &a, const Aabb &b) {
        return Aabb(
                {
                        fminf(a.min.x, b.min.x),
                        fminf(a.min.y, b.min.y),
                        fminf(a.min.z, b.min.z),
                },
                {
                        fmaxf(a.max.x, b.max.x),
                        fmaxf(a.max.y, b.max.y),
                        fmaxf(a.max.z, b.max.z),
                });
}

__host__ Aabb Aabb::from_object(const Object &obj) {
        Aabb bbox = Aabb();
        switch (obj.type) {
                case SPHERE: {
                        Vec3 center0 = obj.sphere.center.orig;
                        Vec3 center1 = obj.sphere.center.orig + obj.sphere.center.dir;

                        Vec3 min0 = center0 - Vec3(obj.sphere.radius);
                        Vec3 max0 = center0 + Vec3(obj.sphere.radius);
                        Vec3 min1 = center1 - Vec3(obj.sphere.radius);
                        Vec3 max1 = center1 + Vec3(obj.sphere.radius);
                        bbox      = Aabb{Vec3(fminf(min0.x, min1.x), fminf(min0.y, min1.y), fminf(min0.z, min1.z)),
                                    Vec3(fmaxf(max0.x, max1.x), fmaxf(max0.y, max1.y), fmaxf(max0.z, max1.z))};
                        break;
                }
                case QUAD: {
                        Vec3 corners[4] = {obj.quad.q, obj.quad.q + obj.quad.u, obj.quad.q + obj.quad.v,
                                           obj.quad.q + obj.quad.u + obj.quad.v};
                        Vec3 min_p      = corners[0];
                        Vec3 max_p      = corners[0];
                        for (int i = 1; i < 4; i++) {
                                min_p.x = fminf(min_p.x, corners[i].x);
                                min_p.y = fminf(min_p.y, corners[i].y);
                                min_p.z = fminf(min_p.z, corners[i].z);
                                max_p.x = fmaxf(max_p.x, corners[i].x);
                                max_p.y = fmaxf(max_p.y, corners[i].y);
                                max_p.z = fmaxf(max_p.z, corners[i].z);
                        }
                        bbox = Aabb(min_p, max_p);
                        break;
                }
        }

        if (obj.rotation != 0) {
                Vec3 min_p      = bbox.min;
                Vec3 max_p      = bbox.max;
                Vec3 corners[8] = {Vec3(min_p.x, min_p.y, min_p.z), Vec3(min_p.x, min_p.y, max_p.z),
                                   Vec3(min_p.x, max_p.y, min_p.z), Vec3(min_p.x, max_p.y, max_p.z),
                                   Vec3(max_p.x, min_p.y, min_p.z), Vec3(max_p.x, min_p.y, max_p.z),
                                   Vec3(max_p.x, max_p.y, min_p.z), Vec3(max_p.x, max_p.y, max_p.z)};

                Vec3 new_min = Vec3(INFINITY, INFINITY, INFINITY);
                Vec3 new_max = Vec3(-INFINITY, -INFINITY, -INFINITY);

                // Rotate corners.
                for (int i = 0; i < 8; i++) {
                        Vec3 rel = corners[i] - obj.rotation_center;

                        // World -> object.
                        float new_x = obj.cos_theta * rel.x + obj.sin_theta * rel.z;
                        float new_z = -obj.sin_theta * rel.x + obj.cos_theta * rel.z;

                        // Object -> world.
                        Vec3 rotated = Vec3(new_x, rel.y, new_z) + obj.rotation_center;

                        for (int c = 0; c < 3; ++c) {
                                new_min[c] = fminf(new_min[c], rotated[c]);
                                new_max[c] = fmaxf(new_max[c], rotated[c]);
                        }
                }
                bbox = Aabb(new_min, new_max);
        }

        bbox += obj.translation;
        return bbox;
}

__host__ void Aabb::pad() {
        const float delta = 0.001f;
        for (int axis = 0; axis < 3; ++axis) {
                float size = max[axis] - min[axis];
                if (size < delta) {
                        float midpoint = 0.5f * (min[axis] + max[axis]);
                        min[axis]      = midpoint - 0.5f * delta;
                        max[axis]      = midpoint + 0.5f * delta;
                }
        }
}
