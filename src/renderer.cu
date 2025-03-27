#include "png_output.cuh"
#include "renderer.cuh"

#include "camera.cuh"
#include "utils.cuh"

#include <GL/gl.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdio.h>
#include <surface_types.h>
#include <time.h>

Renderer::Renderer()
    : gl_texture(0), cuda_texture_resource(nullptr), accumulation_buffer(nullptr), d_scene(nullptr),
      render_needs_update(true), scene_needs_update(true), final_resolution_idx(1) {}

Renderer::~Renderer() {
        if (accumulation_buffer)   cudaFree(accumulation_buffer);
        if (d_scene)               cudaFree(d_scene);
        if (cuda_texture_resource) cudaFree(cuda_texture_resource);
        if (gl_texture)            glDeleteTextures(1, &gl_texture);
}

__device__ Vec3 ray_color(const Ray &r_in, const Scene *scene, RandState *state, int max_depth) {
        Vec3 attenuation(1.0f, 1.0f, 1.0f);
        Vec3 color_acc(0.0f, 0.0f, 0.0f);
        Ray current_ray = r_in;

        for (int depth = 0; depth < max_depth; depth++) {
                HitRecord rec;

                if (scene->hit(current_ray, 0.001f, FMAX, rec)) {
                        Ray scattered;
                        Vec3 scatter_attenuation;
                        if (scene->materials[rec.material_id].scatter(current_ray, rec, scatter_attenuation, scattered,
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

__global__ void render_frame_kernel(float4 *accumulation_buffer, cudaSurfaceObject_t output_surf, RenderConfig config,
                              Camera cam, Scene *scene, int sample_count, int seed, int width, int height) {
        // calculate pixel coordinates
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        // exit if out of bounds
        if (x >= width || y >= height) return;

        // compute 1d index for accumulation buffer
        const int idx = y * width + x;

        // initialize random state for this pixel
        RandState rand_state;
        random_init(&rand_state, seed, idx);

        const float u = float(x + random_float(&rand_state)) / float(width - 1);
        const float v = float(width - 1 - y + random_float(&rand_state)) / float(height - 1);
        Ray ray = cam.get_ray(u, v, &rand_state);

        Vec3 current_color = ray_color(ray, scene, &rand_state, config.max_depth);

        float4 prev_accum        = accumulation_buffer[idx];
        accumulation_buffer[idx] = make_float4(prev_accum.x + current_color.x, prev_accum.y + current_color.y,
                                               prev_accum.z + current_color.z, 1.0f);

        // compute average and apply gamma correction for display
        Vec3 accum_color     = Vec3(accumulation_buffer[idx].x, accumulation_buffer[idx].y, accumulation_buffer[idx].z);
        Vec3 avg_color       = accum_color / float(sample_count);
        Vec3 gamma_corrected = Vec3(sqrtf(avg_color.x), sqrtf(avg_color.y), sqrtf(avg_color.z));

        // convert to uchar4 for texture output (with gamma-corrected color)
        unsigned char r = static_cast<unsigned char>(clamp(gamma_corrected.x, 0.0f, 1.0f) * 255.99f);
        unsigned char g = static_cast<unsigned char>(clamp(gamma_corrected.y, 0.0f, 1.0f) * 255.99f);
        unsigned char b = static_cast<unsigned char>(clamp(gamma_corrected.z, 0.0f, 1.0f) * 255.99f);
        uchar4 pixel    = make_uchar4(r, g, b, 255);

        // write pixel to surface, flipping y to match opengl's coordinate system
        surf2Dwrite(pixel, output_surf, x * sizeof(uchar4), height - 1 - y, cudaBoundaryModeClamp);
}

void Renderer::init(const int width, const int height, RenderConfig render_config) {
        window_w = width;
        window_h = height;
        config   = render_config;

        sample_count = 0;

        size_t buffer_size = window_w * window_h * sizeof(Vec3);
        CHECK_CUDA_ERROR(cudaMalloc(&accumulation_buffer, buffer_size));
        CHECK_CUDA_ERROR(cudaMemset(accumulation_buffer, 0, buffer_size));

        glGenTextures  (1, &gl_texture);
        glBindTexture  (GL_TEXTURE_2D, gl_texture);
        glTexImage2D   (GL_TEXTURE_2D, 0, GL_RGBA8, window_w, window_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture  (GL_TEXTURE_2D, 0);

        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
                fprintf(stderr, "OpenGL error after texture creation: %d\n", err);
                exit(EXIT_FAILURE);
        }

        CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&cuda_texture_resource, gl_texture, GL_TEXTURE_2D,
                                                     cudaGraphicsRegisterFlagsWriteDiscard));
        CHECK_CUDA_ERROR(cudaMalloc(&d_scene, sizeof(Scene)));
}

void Renderer::render_single_frame(const Scene &scene, const Camera &camera) {
        if (render_needs_update) {
                CHECK_CUDA_ERROR(cudaMemset(accumulation_buffer, 0, window_w * window_h * sizeof(float4)));
                sample_count = 0;
                render_needs_update = false;
        }

        // update scene on device if needed
        if (scene_needs_update) {
                CHECK_CUDA_ERROR(cudaMemcpy(d_scene, &scene, sizeof(Scene), cudaMemcpyHostToDevice));
                scene_needs_update = false;
        }

        // map the opengl texture to cuda
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cuda_texture_resource, 0));

        // get the cudaarray_t from the mapped texture resource
        cudaArray_t cuda_array;
        CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_texture_resource, 0, 0));

        // create a surface object from the cudaarray_t
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuda_array;
        cudaSurfaceObject_t output_surf;
        CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&output_surf, &resDesc));

        // define block and grid sizes for kernel launch
        dim3 block_size(16, 16);
        dim3 grid_size((window_w + block_size.x - 1) / block_size.x, (window_h + block_size.y - 1) / block_size.y);
        sample_count++;

        // launch the rendering kernel
        render_frame_kernel<<<grid_size, block_size>>>(accumulation_buffer, output_surf, config, camera, d_scene,
                                                 sample_count, time(NULL), window_w, window_h);

        // check for kernel launch or execution errors
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // clean up the surface object
        CHECK_CUDA_ERROR(cudaDestroySurfaceObject(output_surf));

        // unmap the texture resource
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cuda_texture_resource, 0));
}

__global__ void render_full_kernel(unsigned int *buffer, RenderConfig config, Camera cam, Scene *scene, int seed) {
        // calculate pixel coordinates
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        const int w = config.image_w;
        const int h = config.image_h;

        // exit if out of bounds
        if (x >= w || y >= h) return;

        // compute 1d index for accumulation buffer
        const int idx = y * w + x;

        // initialize random state for this pixel
        RandState rand_state;
        random_init(&rand_state, seed, idx);

        // compute color
        Vec3 color;
        for (int s = 0; s < config.samples_per_pixel; s++) {
                const float u = float(x + random_float(&rand_state)) / float(w - 1);
                const float v = float(w - 1 - y + random_float(&rand_state)) / float(h - 1);
                Ray ray       = cam.get_ray(u, v, &rand_state);

                color += ray_color(ray, scene, &rand_state, config.max_depth);
        }
        color /= float(config.samples_per_pixel);

        // gamma correction
        color = Vec3(sqrtf(color.x), sqrtf(color.y), sqrtf(color.z));

        // convert to uint and insert into buffer
        buffer[idx] = make_color(color.x, color.y, color.z);
}

void Renderer::render_full_frame(const char file_path[], const Camera &camera) {
        switch (final_resolution_idx) {
                case 0 : set_final_resolution(1280,  720); break;
                case 1 : set_final_resolution(1920, 1080); break;
                case 2 : set_final_resolution(2560, 1440); break;
                case 4 : set_final_resolution(3840, 2160); break;
                default: {
                        fprintf(stderr, "Invalid resolution idx: %d", final_resolution_idx);
                        exit(EXIT_FAILURE);
                }
        }

        // allocate device buffer
        unsigned int *d_buffer;
        size_t buffer_size = config.image_w * config.image_w * sizeof(unsigned int);
        CHECK_CUDA_ERROR(cudaMalloc(&d_buffer, buffer_size));
        CHECK_CUDA_ERROR(cudaMemset(d_buffer, 0, buffer_size));

        // define block and grid sizes for kernel launch
        dim3 block_size(16, 16);
        dim3 grid_size((config.image_w + block_size.x - 1) / block_size.x, (config.image_h + block_size.y - 1) / block_size.y);

        render_full_kernel<<<grid_size, block_size>>>(d_buffer, config, camera, d_scene, time(NULL));

        // check for kernel launch or execution errors
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // copy buffer back to host
        unsigned int *h_buffer = new unsigned int [buffer_size];
        CHECK_CUDA_ERROR(cudaMemcpy(h_buffer, d_buffer, buffer_size, cudaMemcpyDeviceToHost));

        // export
        save_png(file_path, h_buffer, config.image_w, config.image_h);
}
