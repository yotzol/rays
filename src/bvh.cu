#include "bvh.cuh"

#include "scene.cuh"

#include <algorithm>

__host__ int Scene::build_bvh_recursive(std::vector<BvhNode> &nodes, std::vector<int> &obj_idxs, int start, int end) {
        BvhNode node;
        if (end - start <= 1) {
                // leaf node
                node.is_leaf        = true;
                node.leaf.count     = 1;
                node.leaf.idx_start = obj_idxs[start];
                node.bbox           = Aabb::from_object(objects[obj_idxs[start]]);
        } else {
                // internal node: split along longest axis
                Vec3 extent = Aabb(objects, obj_idxs, start, end).max - Aabb(objects, obj_idxs, start, end).min;
                int axis    = (extent.x > extent.y && extent.x > extent.z) ? 0 : (extent.y > extent.z ? 1 : 2);

                std::sort(obj_idxs.begin() + start, obj_idxs.begin() + end,
                          [&](int a, int b) { return objects[a].center()[axis] < objects[b].center()[axis]; });
                int mid = start + (end - start) / 2;

                node.is_leaf     = false;
                node.inner.idx_l = build_bvh_recursive(nodes, obj_idxs, start, mid);
                node.inner.idx_r = build_bvh_recursive(nodes, obj_idxs, mid, end);
                node.bbox        = Aabb::merge(nodes[node.inner.idx_l].bbox, nodes[node.inner.idx_r].bbox);
        }
        int node_idx = nodes.size();
        nodes.push_back(node);
        return node_idx;
}

__host__ void Scene::build_bvh() {
        std::vector<BvhNode> nodes;
        std::vector<int> obj_idxs(num_objects);
        for (int i = 0; i < num_objects; i++) obj_idxs[i] = i;

        root_idx = build_bvh_recursive(nodes, obj_idxs, 0, num_objects);
        for (int i = 0; i < nodes.size(); i++) bvh_nodes[i] = nodes[i];
}
