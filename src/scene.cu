#include "scene.cuh"

#include "object.cuh"
#include "utils.cuh"

#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

__host__ void Scene::add_object(Object obj, float rotation, Vec3 translation) {
        assert(num_objects < MAX_OBJECTS);

        obj.translation     = translation;
        float r             = fmod(rotation, 360);
        obj.rotation        = r > 0 ? r : 360 + r;
        obj.rotation_center = obj.center();

        float rad     = radians(obj.rotation);
        obj.sin_theta = sinf(rad);
        obj.cos_theta = cosf(rad);

        objects[num_objects++] = obj;
}

__host__ void Scene::add_object(Object obj, float rotation, Vec3 translation, Vec3 rotation_center) {
        assert(num_objects < MAX_OBJECTS);

        obj.translation     = translation;
        float r             = fmod(rotation, 360);
        obj.rotation        = r > 0 ? r : 360 + r;
        obj.rotation_center = rotation_center;

        float rad           = radians(rotation);
        obj.sin_theta       = sinf(rad);
        obj.cos_theta       = cosf(rad);

        objects[num_objects++] = obj;
}

__host__ int Scene::add_material(const Material &material) {
        assert(num_materials < MAX_MATERIALS);
        materials[num_materials] = material;
        return num_materials++;
}

__host__ void Scene::add_box(const Vec3 a, const Vec3 b, const int material_id, float rotation, Vec3 translation) {
        Vec3 min = Vec3(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
        Vec3 max = Vec3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));

        Vec3 center = (min + max) * 0.5f;

        Vec3 dx = Vec3(max.x - min.x, 0, 0);
        Vec3 dy = Vec3(0, max.y - min.y, 0);
        Vec3 dz = Vec3(0, 0, max.z - min.z);

        add_object(quad_new(Vec3(min.x, min.y, max.z),  dx,  dy, material_id), rotation, translation, center);  // front
        add_object(quad_new(Vec3(max.x, min.y, max.z), -dz,  dy, material_id), rotation, translation, center);  // right
        add_object(quad_new(Vec3(max.x, min.y, min.z), -dx,  dy, material_id), rotation, translation, center);  // back
        add_object(quad_new(Vec3(min.x, min.y, min.z),  dz,  dy, material_id), rotation, translation, center);  // left
        add_object(quad_new(Vec3(min.x, max.y, max.z),  dx, -dz, material_id), rotation, translation, center);  // top
        add_object(quad_new(Vec3(min.x, min.y, min.z),  dx,  dz, material_id), rotation, translation, center);  // bottom
}

__host__ void Scene::set_env_map(const char path[]) {
        // load rgba image
        unsigned char *env_data = stbi_load(path, &env_w, &env_h, &env_channels, 4);
        if (!env_data) {
                fprintf(stderr, "Failed to load environment map: %s\n", path);
                return;
        }

        // rgba channel format
        const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

        // copy to device
        cudaArray *d_array;
        CHECK_CUDA_ERROR(cudaMallocArray(&d_array, &channel_desc, env_w, env_h));
        const int array_size = env_w * env_h * 4 * sizeof(unsigned char);
        CHECK_CUDA_ERROR(cudaMemcpyToArray(d_array, 0, 0, env_data, array_size, cudaMemcpyHostToDevice));

        // resource descriptor
        cudaResourceDesc res_desc{};
        res_desc.resType         = cudaResourceTypeArray;
        res_desc.res.array.array = d_array;

        // texture descriptor
        cudaTextureDesc tex_desc{};
        tex_desc.addressMode[0]   = cudaAddressModeWrap;          // wrap horizontally
        tex_desc.addressMode[1]   = cudaAddressModeClamp;         // clamp vertically
        tex_desc.filterMode       = cudaFilterModeLinear;         // linear interpolation
        tex_desc.readMode         = cudaReadModeNormalizedFloat;  // return normalized floats [0,1]
        tex_desc.normalizedCoords = 1;                            // use [0,1] coordinates

        // texture object
        CHECK_CUDA_ERROR(cudaCreateTextureObject(&env_map, &res_desc, &tex_desc, nullptr));

        // free host memory
        stbi_image_free(env_data);
}

__device__ bool Scene::hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const {
        int stack[32];  // fixed-size stack
        int stack_ptr        = 0;
        stack[stack_ptr++]   = root_idx;
        bool hit_anything    = false;
        float closest_so_far = t_max;
        HitRecord temp_rec;

        while (stack_ptr > 0) {
                int node_idx        = stack[--stack_ptr];
                const BvhNode &node = bvh_nodes[node_idx];

                if (node.bbox.hit(ray, t_min, t_max)) {
                        if (node.is_leaf) {
                                for (int i = 0; i < node.leaf.count; i++) {
                                        int obj_idx = node.leaf.idx_start + i;
                                        if (objects[obj_idx].hit(ray, t_min, closest_so_far, temp_rec)) {
                                                hit_anything   = true;
                                                closest_so_far = temp_rec.t;
                                                rec            = temp_rec;
                                        }
                                }
                        } else {
                                // internal: push children
                                stack[stack_ptr++] = node.inner.idx_l;
                                stack[stack_ptr++] = node.inner.idx_r;
                        }
                }
        }
        return hit_anything;
}
