#include "renderer.hpp"

#include "camera.hpp"
#include "image.hpp"
#include "utils.hpp"
#include "vec3.hpp"

#include <GL/gl.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <stdio.h>
#include <time.h>

Renderer::Renderer()
    : gl_texture(0), cuda_texture_resource(nullptr), accumulation_buffer(nullptr), d_scene(nullptr),
      render_needs_update(true), scene_needs_update(true), final_resolution_idx(1),
      camera_positions(std::vector<Camera>()) {}

Renderer::~Renderer() {
        if (accumulation_buffer) cudaFree(accumulation_buffer);
        if (d_scene) cudaFree(d_scene);
        if (cuda_texture_resource) cudaFree(cuda_texture_resource);
        if (gl_texture) glDeleteTextures(1, &gl_texture);
}

__device__ bool Aabb::hit(const Ray &ray, float t_min, float t_max) const {
        for (int axis = 0; axis < 3; ++axis) {
                // Extract components for this axis.
                const float origin = ray.orig[axis];
                const float dir    = ray.dir[axis];
                const float min_b  = min[axis];
                const float max_b  = max[axis];

                // Compute intersection t-values for this slab.
                const float inv_dir = 1.0f / dir;  // Inverse direction for efficiency.
                float t0            = (min_b - origin) * inv_dir;
                float t1            = (max_b - origin) * inv_dir;

                // Swap t0 and t1 if direction is negative.
                if (inv_dir < 0.0f) {
                        float temp = t0;
                        t0         = t1;
                        t1         = temp;
                }

                // Update interval.
                t_min = t0 > t_min ? t0 : t_min;
                t_max = t1 < t_max ? t1 : t_max;

                // If t_min exceeds t_max, no intersection is possible.
                if (t_max <= t_min) return false;
        }
        return true;
}

__device__ bool Scene::hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const {
        int stack[32];  // Fixed-size stack.
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
                                // Internal: push children.
                                stack[stack_ptr++] = node.inner.idx_l;
                                stack[stack_ptr++] = node.inner.idx_r;
                        }
                }
        }
        return hit_anything;
}

__device__ Vec3 ray_color(const Ray &r_in, const Scene *scene, curandState *state, int max_depth) {
        Vec3 throughput(1.0f);
        Vec3 color_acc(0.0f);
        Ray current_ray = r_in;

        for (int depth = 0; depth < max_depth; depth++) {
                HitRecord rec;

                if (scene->hit(current_ray, 0.001f, FMAX, rec)) {
                        Vec3 emission = scene->materials[rec.material_id].emitted(rec.u, rec.v, rec.point);
                        color_acc += throughput * emission;

                        Ray scattered;
                        Vec3 scatter_attenuation;
                        if (scene->materials[rec.material_id].scatter(current_ray, rec, scatter_attenuation, scattered,
                                                                      state)) {
                                // Update attenuation for this bounce.
                                throughput  = throughput * scatter_attenuation;
                                current_ray = scattered;
                        } else {
                                // Ray was absorbed.
                                break;
                        }
                } else {
                        // Hit nothing. Return environment.
                        if (scene->no_environment) {
                                color_acc += throughput * scene->background_color;
                                break;
                        }

                        Vec3 background;
                        // Environment map loaded: use map.
                        if (scene->env_map != 0) {
                                Vec3 d       = normalize(current_ray.dir);
                                float phi    = atan2f(d.z, d.x);
                                float u      = (phi / (2.0f * M_PI)) + 0.5f;
                                float theta  = acosf(d.y);
                                float v      = theta / M_PI;
                                float4 color = tex2D<float4>(scene->env_map, u, v);
                                background   = Vec3(color.x, color.y, color.z);
                        }
                        // No environment loaded: use sky gradient.
                        else {
                                Vec3 unit_direction = normalize(current_ray.dir);
                                float t             = 0.5f * (unit_direction.y + 1.0f);
                                background          = Vec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + Vec3(0.5f, 0.7f, 1.0f) * t;
                        }
                        color_acc += throughput * background;
                        break;
                }

                // Early termination for low-contribution paths.
                float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
                if (randf(state) >= p) {
                        break;
                }
                throughput = throughput / p;
        }

        // Hit max depth. return black.
        return color_acc;
}

__global__ void render_frame_kernel(float4 *accumulation_buffer, cudaSurfaceObject_t output_surf, RenderConfig config,
                                    Camera cam, Scene *scene, int sample_count, int seed, int width, int height) {
        // Calculate pixel coordinates.
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        // Exit if out of bounds.
        if (x >= width || y >= height) return;

        // Compute 1d index for accumulation buffer.
        const int idx = y * width + x;

        // Initialize random state for this pixel.
        curandState rand_state;
        curand_init(seed, idx, 0, &rand_state);

        const float u = float(x + randf(&rand_state)) / float(width - 1);
        const float v = float(width - 1 - y + randf(&rand_state)) / float(height - 1);
        Ray ray       = cam.get_ray(u, v, &rand_state);

        Vec3 current_color = ray_color(ray, scene, &rand_state, config.max_depth);

        float4 prev_accum        = accumulation_buffer[idx];
        accumulation_buffer[idx] = make_float4(prev_accum.x + current_color.x, prev_accum.y + current_color.y,
                                               prev_accum.z + current_color.z, 1.0f);

        // Compute average and apply gamma correction for display.
        Vec3 accum_color     = Vec3(accumulation_buffer[idx].x, accumulation_buffer[idx].y, accumulation_buffer[idx].z);
        Vec3 avg_color       = accum_color / float(sample_count);
        Vec3 gamma_corrected = Vec3(sqrtf(avg_color.x), sqrtf(avg_color.y), sqrtf(avg_color.z));

        // Convert to uchar4 for texture output (with gamma-corrected color).
        unsigned char r = static_cast<unsigned char>(clamp(gamma_corrected.x, 0.0f, 1.0f) * 255.99f);
        unsigned char g = static_cast<unsigned char>(clamp(gamma_corrected.y, 0.0f, 1.0f) * 255.99f);
        unsigned char b = static_cast<unsigned char>(clamp(gamma_corrected.z, 0.0f, 1.0f) * 255.99f);
        uchar4 pixel    = make_uchar4(r, g, b, 255);

        // Write pixel to surface, flipping y to match OpenGL's coordinate system.
        surf2Dwrite(pixel, output_surf, x * sizeof(uchar4), height - 1 - y, cudaBoundaryModeClamp);
}

void Renderer::init(RenderConfig render_config) {
        config = render_config;

        sample_count = 0;

        size_t buffer_size = config.window_w * config.window_h * sizeof(Vec3);
        CHECK_CUDA_ERROR(cudaMalloc(&accumulation_buffer, buffer_size));
        CHECK_CUDA_ERROR(cudaMemset(accumulation_buffer, 0, buffer_size));

        glGenTextures(1, &gl_texture);
        glBindTexture(GL_TEXTURE_2D, gl_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, config.window_w, config.window_h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);

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
        // Reset accumulation buffer if camera moved.
        if (render_needs_update) {
                CHECK_CUDA_ERROR(
                        cudaMemset(accumulation_buffer, 0, config.window_w * config.window_h * sizeof(float4)));
                sample_count = 0;
                render_needs_update = false;
        }

        // Update scene on device if needed.
        if (scene_needs_update) {
                CHECK_CUDA_ERROR(cudaMemcpy(d_scene, &scene, sizeof(Scene), cudaMemcpyHostToDevice));
                scene_needs_update = false;
        }

        // Map the OpenGL texture to CUDA.
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cuda_texture_resource, 0));

        // Get CUDA array from the mapped texture resource.
        cudaArray_t cuda_array;
        CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_texture_resource, 0, 0));

        // Create a surface object from the CUDA array.
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuda_array;
        cudaSurfaceObject_t output_surf;
        CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&output_surf, &resDesc));

        // Define block and grid sizes for kernel launch.
        dim3 block_size(16, 16);
        dim3 grid_size((config.window_w + block_size.x - 1) / block_size.x,
                       (config.window_h + block_size.y - 1) / block_size.y);
        sample_count++;

        // Launch the rendering kernel.
        render_frame_kernel<<<grid_size, block_size>>>(accumulation_buffer, output_surf, config, camera, d_scene,
                                                       sample_count, time(NULL), config.window_w, config.window_h);

        // Check for kernel launch or execution errors.
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Clean up the surface object.
        CHECK_CUDA_ERROR(cudaDestroySurfaceObject(output_surf));

        // Unmap the texture resource.
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cuda_texture_resource, 0));
}

__global__ void render_full_kernel(unsigned int *buffer, RenderConfig config, Camera cam, Scene *scene, int seed) {
        // Calculate pixel coordinates.
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        const int w = config.image_w;
        const int h = config.image_h;

        // Exit if out of bounds.
        if (x >= w || y >= h) return;

        // Compute 1d index for accumulation buffer.
        const int idx = y * w + x;

        // Initialize random state for this pixel.
        curandState rand_state;
        curand_init(seed, idx, 0, &rand_state);

        // Compute color.
        Vec3 color;
        for (int s = 0; s < config.samples_per_pixel; s++) {
                const float u = float(x + randf(&rand_state)) / float(w - 1);
                const float v = float(w - 1 - y + randf(&rand_state)) / float(h - 1);
                Ray ray       = cam.get_ray(u, v, &rand_state);

                color += ray_color(ray, scene, &rand_state, config.max_depth);
        }
        color /= float(config.samples_per_pixel);

        // Gamma correction.
        color = Vec3(sqrtf(color.x), sqrtf(color.y), sqrtf(color.z));

        // Convert to uint and insert into buffer.
        buffer[idx] = make_color(color.x, color.y, color.z);
}

void Renderer::render_full_frame(const char file_path[], const Camera &camera) {
        switch (final_resolution_idx) {
                case 0 : set_final_resolution(640, 360); break;
                case 1 : set_final_resolution(1280, 720); break;
                case 2 : set_final_resolution(1920, 1080); break;
                case 3 : set_final_resolution(2560, 1440); break;
                case 4 : set_final_resolution(3840, 2160); break;
                case 5 : set_final_resolution(1080, 1080); break;
                default: {
                        fprintf(stderr, "Invalid resolution idx: %d", final_resolution_idx);
                        exit(EXIT_FAILURE);
                }
        }

        // Allocate device buffer.
        unsigned int *d_buffer;
        size_t buffer_size = config.image_w * config.image_w * sizeof(unsigned int);
        CHECK_CUDA_ERROR(cudaMalloc(&d_buffer, buffer_size));
        CHECK_CUDA_ERROR(cudaMemset(d_buffer, 0, buffer_size));

        // Define block and grid sizes for kernel launch.
        dim3 block_size(16, 16);
        dim3 grid_size((config.image_w + block_size.x - 1) / block_size.x,
                       (config.image_h + block_size.y - 1) / block_size.y);

        render_full_kernel<<<grid_size, block_size>>>(d_buffer, config, camera, d_scene, time(NULL));

        // Check for kernel launch or execution errors.
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Copy buffer back to host.
        unsigned int *h_buffer = new unsigned int[buffer_size];
        CHECK_CUDA_ERROR(cudaMemcpy(h_buffer, d_buffer, buffer_size, cudaMemcpyDeviceToHost));

        // Export.
        save_png(file_path, h_buffer, config.image_w, config.image_h);

        printf("Export complete: %s.\n", file_path);
}

static const int frames_per_transition = 120;

namespace fs = std::filesystem;

bool create_directory(const char path[]) {
        try {
                if (!fs::exists(path)) {
                        fs::create_directory(path);
                        printf("Created directory: %s\n", path);
                } else {
                        printf("Directory already exists: %s\n", path);
                }
                return true;
        } catch (const fs::filesystem_error &e) {
                fprintf(stderr, "Error creating directory: %s", e.what());
                return false;
        }
}

void Renderer::render_video(const char name[]) {
        if (camera_positions.size() < 2) {
                fprintf(stderr, "Need at least two camera positions for video rendering.\n");
                return;
        }

        int n            = camera_positions.size();
        int total_frames = (n - 1) * frames_per_transition;
        char frame_name[256];

        if (!create_directory(name)) return;

        for (int frame_idx = 0; frame_idx < total_frames; frame_idx++) {
                // Determine indexes.
                int trans_idx = frame_idx / frames_per_transition;
                int local_idx = frame_idx % frames_per_transition;

                float t = (float)local_idx / (frames_per_transition - 1);  // T from 0 to 1.

                const Camera &cam_a = camera_positions[trans_idx];
                const Camera &cam_b = camera_positions[trans_idx + 1];

                // Create new camera for this frame.
                Camera cam_interp;
                cam_interp.aspect_ratio = cam_a.aspect_ratio;

                // Interpolate.
                cam_interp.origin     = lerp(cam_a.origin, cam_b.origin, t);
                cam_interp.yaw        = lerp(cam_a.yaw, cam_b.yaw, t);
                cam_interp.pitch      = lerp(cam_a.pitch, cam_b.pitch, t);
                cam_interp.roll       = lerp(cam_a.roll, cam_b.roll, t);
                cam_interp.vfov       = lerp(cam_a.vfov, cam_b.vfov, t);
                cam_interp.aperture   = lerp(cam_a.aperture, cam_b.aperture, t);
                cam_interp.focus_dist = lerp(cam_a.focus_dist, cam_b.focus_dist, t);

                cam_interp.update_camera();

                // Render the frame.
                snprintf(frame_name, sizeof(frame_name), "%s/frame_%05d.png", name, frame_idx);
                render_full_frame(frame_name, cam_interp);
        }

        printf("Video rendering complete: %s\n", name);
}
