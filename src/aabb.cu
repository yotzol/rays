#include "aabb.cuh"

__host__ Aabb::Aabb(const Object *objects, const std::vector<int> &idxs, int start, int end) {
        if (start >= end) {
                min = Vec3(0.0f, 0.0f, 0.0f);
                max = Vec3(0.0f, 0.0f, 0.f);
                return;
        }

        Aabb box = Aabb::from_object(objects[idxs[start]]);
        for (int i = start + 1; i < end; ++i) {
                Aabb obj_box = Aabb::from_object(objects[idxs[i]]);
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
        switch (obj.type) {
                case SPHERE: {
                        Vec3 center0 = obj.sphere.center.origin;
                        Vec3 center1 = obj.sphere.center.origin + obj.sphere.center.direction;

                        Vec3 min0 = center0 - Vec3(obj.sphere.radius);
                        Vec3 max0 = center0 + Vec3(obj.sphere.radius);
                        Vec3 min1 = center1 - Vec3(obj.sphere.radius);
                        Vec3 max1 = center1 + Vec3(obj.sphere.radius);
                        return Aabb{Vec3(fminf(min0.x, min1.x), fminf(min0.y, min1.y), fminf(min0.z, min1.z)),
                                    Vec3(fmaxf(max0.x, max1.x), fmaxf(max0.y, max1.y), fmaxf(max0.z, max1.z))};
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
                        return Aabb(min_p, max_p);
                }
        }
        return Aabb();
}

__device__ bool Aabb::hit(const Ray &ray, float t_min, float t_max) const {
        for (int axis = 0; axis < 3; ++axis) {
                // extract components for this axis
                float origin = ray.origin[axis];
                float dir    = ray.direction[axis];
                float min_b  = min[axis];
                float max_b  = max[axis];

                // compute intersection t-values for this slab
                float inv_dir = 1.0f / dir;  // inverse direction for efficiency
                float t0      = (min_b - origin) * inv_dir;
                float t1      = (max_b - origin) * inv_dir;

                // swap t0 and t1 if direction is negative
                if (inv_dir < 0.0f) {
                        float temp = t0;
                        t0         = t1;
                        t1         = temp;
                }

                // update interval
                t_min = t0 > t_min ? t0 : t_min;
                t_max = t1 < t_max ? t1 : t_max;

                // if t_min exceeds t_max, no intersection is possible
                if (t_max <= t_min) return false;
        }
        return true;
}

__host__ void Aabb::pad() {
        const float delta = 0.0001f;
        for (int axis = 0; axis < 3; ++axis) {
                float size = max[axis] - min[axis];
                if (size < delta) {
                        float midpoint = 0.5f * (min[axis] + max[axis]);
                        min[axis]      = midpoint - 0.5f * delta;
                        max[axis]      = midpoint + 0.5f * delta;
                }
        }
}
