#ifndef RENDER_CUH
#define RENDER_CUH

#include "camera.cuh"
#include "scene.cuh"
#include "utils.cuh"

#include <ctime>
#include <stdio.h>

// render configuration
struct RenderConfig {
        int image_w;
        int image_h;
        int samples_per_pixel;
        int max_depth;
};

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call)                                                                                         \
        {                                                                                                              \
                cudaError_t err = call;                                                                                \
                if (err != cudaSuccess) {                                                                              \
                        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__,                       \
                                cudaGetErrorString(err));                                                              \
                        exit(EXIT_FAILURE);                                                                            \
                }                                                                                                      \
        }

// color function to trace rays through the scene
__device__ Vec3 ray_color(const Ray &r_in, const Scene *world, RandState *state, int max_depth) {
        Vec3 attenuation(1.0f, 1.0f, 1.0f);
        Vec3 color_acc(0.0f, 0.0f, 0.0f);
        Ray current_ray = r_in;

        for (int depth = 0; depth < max_depth; depth++) {
                HitRecord rec;

                if (world->hit(current_ray, 0.001f, FMAX, rec)) {
                        Ray scattered;
                        Vec3 scatter_attenuation;

                        if (world->materials[rec.material_id].scatter(current_ray, rec, scatter_attenuation, scattered,
                                                                      state)) {
                                // update attenuation for this bounce
                                attenuation = attenuation * scatter_attenuation;
                                current_ray = scattered;
                        } else {
                                // ray was absorbed. return black
                                return Vec3(0, 0, 0);
                        }
                } else {
                        // if it hit nothing, it's the sky. return blue white gradient
                        Vec3 unit_direction = normalize(current_ray.direction);
                        float t             = 0.5f * (unit_direction.y + 1.0f);
                        Vec3 background     = Vec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + Vec3(0.5f, 0.7f, 1.0f) * t;
                        return attenuation * background;
                }

                // early termination for low-contribution paths
                float p = fmaxf(attenuation.x, fmaxf(attenuation.y, attenuation.z));
                if (random_float(state) >= p) {
                        break;
                }
                attenuation = attenuation / p;
        }

        // hit max depth. return black
        return Vec3(0, 0, 0);
}

// main rendering kernel
__global__ void render_kernel(unsigned int *framebuffer, RenderConfig config, Camera cam, Scene *world, int seed) {
        // calculate pixel position
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // check if this thread is within the image bounds
        if (x >= config.image_w || y >= config.image_h) return;

        // calculate buffer index
        int idx = y * config.image_w + x;

        // initialize random state with unique seed per thread
        RandState rand_state;
        random_init(&rand_state, seed, idx);

        Vec3 color(0, 0, 0);

        // accumulate samples
        for (int s = 0; s < config.samples_per_pixel; s++) {
                // calculate u,v coordinates in [0,1] with jitter for antialiasing
                float u = float(x + random_float(&rand_state)) / float(config.image_w - 1);
                float v = float(config.image_h - 1 - y + random_float(&rand_state)) / float(config.image_h - 1);

                // trace ray and accumulate color
                Ray ray = cam.get_ray(u, v, &rand_state);
                color   = color + ray_color(ray, world, &rand_state, config.max_depth);
        }

        // average samples and apply gamma correction
        color = color / float(config.samples_per_pixel);
        color = Vec3(sqrtf(color.x), sqrtf(color.y), sqrtf(color.z));

        // write final color to framebuffer
        framebuffer[idx] = make_color(color.x, color.y, color.z);
}

// wrapper function to set up and launch the kernel
void render_scene(unsigned int *host_framebuffer, int width, int height, int samples, int max_depth, Camera &camera,
                  Scene &scene) {
        // device memory
        unsigned int *device_framebuffer;
        Scene *device_scene;
        RenderConfig config;

        // Set up config
        config.image_w           = width;
        config.image_h           = height;
        config.samples_per_pixel = samples;
        config.max_depth         = max_depth;

        // allocate device framebuffer
        size_t framebuffer_size = width * height * sizeof(unsigned int);
        CHECK_CUDA_ERROR(cudaMalloc(&device_framebuffer, framebuffer_size));

        // initialize framebuffer with white
        CHECK_CUDA_ERROR(cudaMemset(device_framebuffer, 0xFF, framebuffer_size));

        // allocate and copy scene to device memory
        CHECK_CUDA_ERROR(cudaMalloc(&device_scene, sizeof(Scene)));
        CHECK_CUDA_ERROR(cudaMemcpy(device_scene, &scene, sizeof(Scene), cudaMemcpyHostToDevice));

        // calculate grid and block dimensions
        dim3 block_size(16, 16);  // 256 threads per block
        dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

        printf("Launching kernel with grid size (%d, %d) and block size (%d, %d)\n", grid_size.x, grid_size.y,
               block_size.x, block_size.y);

        // launch rendering kernel
        render_kernel<<<grid_size, block_size>>>(device_framebuffer, config, camera, device_scene,
                                                 time(NULL)  // random seed based on current time
        );

        // check for kernel launch errors
        CHECK_CUDA_ERROR(cudaGetLastError());

        // wait for kernel to finish
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // copy result back to host
        CHECK_CUDA_ERROR(cudaMemcpy(host_framebuffer, device_framebuffer, framebuffer_size, cudaMemcpyDeviceToHost));

        // free device memory
        CHECK_CUDA_ERROR(cudaFree(device_framebuffer));
        CHECK_CUDA_ERROR(cudaFree(device_scene));
}

#endif  // RENDER_CUH
