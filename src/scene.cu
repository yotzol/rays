#include "scene.cuh"

#include <assert.h>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

__host__ void Scene::add_object(const Object &obj) {
        if (num_objects < MAX_OBJECTS) {
                objects[num_objects++] = obj;
        }
        assert(num_objects < MAX_OBJECTS);
}

__host__ int Scene::add_material(const Material &material) {
        if (num_materials < MAX_MATERIALS) {
                materials[num_materials] = material;
                return num_materials++;
        }
        assert(num_materials < MAX_MATERIALS);
        return -1;
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
