#include "scene.hpp"

#include "image.hpp"
#include "object.hpp"
#include "utils.hpp"

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstring>

void Scene::add_object(Object obj, float rotation, Vec3 translation) {
        assert(num_objects < (int)MAX_OBJECTS);

        obj.translation     = translation;
        float r             = fmodf(rotation, 360);
        obj.rotation        = r > 0 ? r : 360 + r;
        obj.rotation_center = obj.center();

        float rad     = to_radians(obj.rotation);
        obj.sin_theta = sinf(rad);
        obj.cos_theta = cosf(rad);

        objects[num_objects++] = obj;
}

void Scene::add_object(Object obj, float rotation, Vec3 translation, Vec3 rotation_center) {
        assert(num_objects < (int)MAX_OBJECTS);

        obj.translation     = translation;
        float r             = fmodf(rotation, 360);
        obj.rotation        = r > 0 ? r : 360 + r;
        obj.rotation_center = rotation_center;

        float rad     = to_radians(rotation);
        obj.sin_theta = sinf(rad);
        obj.cos_theta = cosf(rad);

        objects[num_objects++] = obj;
}

int Scene::add_material(const Material &material) {
        assert(num_materials < (int)MAX_MATERIALS);
        materials[num_materials] = material;
        return num_materials++;
}

void Scene::add_box(const Vec3 a, const Vec3 b, const int material_id, float rotation, Vec3 translation) {
        Vec3 min = Vec3(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
        Vec3 max = Vec3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));

        Vec3 center = (min + max) * 0.5f;

        Vec3 dx = Vec3(max.x - min.x, 0, 0);
        Vec3 dy = Vec3(0, max.y - min.y, 0);
        Vec3 dz = Vec3(0, 0, max.z - min.z);

        add_object(quad_new(Vec3(min.x, min.y, max.z), dx, dy, material_id), rotation, translation, center);   // front
        add_object(quad_new(Vec3(max.x, min.y, max.z), -dz, dy, material_id), rotation, translation, center);  // right
        add_object(quad_new(Vec3(max.x, min.y, min.z), -dx, dy, material_id), rotation, translation, center);  // back
        add_object(quad_new(Vec3(min.x, min.y, min.z), dz, dy, material_id), rotation, translation, center);   // left
        add_object(quad_new(Vec3(min.x, max.y, max.z), dx, -dz, material_id), rotation, translation, center);  // top
        add_object(quad_new(Vec3(min.x, min.y, min.z), dx, dz, material_id), rotation, translation, center);   // bottom
}

void Scene::set_env_map(const char filename[]) {
        env_map = load_texture(filename);
}

int Scene::build_bvh_recursive(std::vector<BvhNode> &nodes, std::vector<int> &obj_idxs, int start, int end) {
        BvhNode node;
        if (end - start <= 1) {
                // Leaf node.
                node.is_leaf        = true;
                node.leaf.count     = 1;
                node.leaf.idx_start = obj_idxs[(size_t)start];
                node.bbox           = Aabb::from_object(objects[obj_idxs[(size_t)start]]);
        } else {
                // Internal node: split along longest axis.
                Vec3 extent = Aabb(objects, obj_idxs, start, end).max - Aabb(objects, obj_idxs, start, end).min;
                int axis    = (extent.x > extent.y && extent.x > extent.z) ? 0 : (extent.y > extent.z ? 1 : 2);

                std::sort(obj_idxs.begin() + start, obj_idxs.begin() + end,
                          [&](int a, int b) { return objects[a].center()[axis] < objects[b].center()[axis]; });
                int mid = start + (end - start) / 2;

                node.is_leaf     = false;
                node.inner.idx_l = build_bvh_recursive(nodes, obj_idxs, start, mid);
                node.inner.idx_r = build_bvh_recursive(nodes, obj_idxs, mid, end);
                node.bbox = Aabb::merge(nodes[(size_t)node.inner.idx_l].bbox, nodes[(size_t)node.inner.idx_r].bbox);
        }
        int node_idx = (int)nodes.size();
        nodes.push_back(node);
        return node_idx;
}

void Scene::build_bvh() {
        std::vector<BvhNode> nodes;
        std::vector<int> obj_idxs((size_t)num_objects);
        for (size_t i = 0; i < (size_t)num_objects; i++) obj_idxs[i] = (int)i;

        root_idx = build_bvh_recursive(nodes, obj_idxs, 0, num_objects);
        for (size_t i = 0; i < nodes.size(); i++) bvh_nodes[i] = nodes[i];
}

Object sphere_new(const Vec3 static_center, float radius, const int material_id) {
        Object obj;

        obj.type            = SPHERE;
        obj.material_id     = material_id;
        obj.sphere.center   = Ray(static_center, Vec3(0.0f, 0.0f, 0.0f));
        obj.sphere.radius   = std::fmax(radius, 0.0f);
        obj.rotation_center = static_center;

        return obj;
}

Object sphere_moving(const Vec3 &center1, const Vec3 &center2, float radius, const int material_id) {
        Object obj;

        obj.type            = SPHERE;
        obj.material_id     = material_id;
        obj.sphere.center   = Ray(center1, center2 - center1);
        obj.sphere.radius   = std::fmax(radius, 0.0f);
        obj.rotation_center = center1;

        return obj;
}

Object quad_new(const Vec3 q, const Vec3 u, const Vec3 v, const int material_id) {
        Object obj;
        obj.type        = QUAD;
        obj.material_id = material_id;
        obj.quad.q      = q;
        obj.quad.u      = u;
        obj.quad.v      = v;

        Vec3 n          = cross(u, v);
        obj.quad.normal = normalize(n);
        obj.quad.d      = dot(obj.quad.normal, q);
        obj.quad.w      = n / dot(n, n);

        return obj;
}
