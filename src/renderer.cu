#include "renderer.cuh"

#include "camera.cuh"
#include "utils.cuh"

#include <GL/gl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <surface_types.h>
#include <time.h>

Renderer::Renderer()
    : gl_texture(0), cuda_texture_resource(nullptr), accumulation_buffer(nullptr), display_buffer(nullptr),
      d_scene(nullptr), scene_needs_update(true) {}

Renderer::~Renderer() {
        if (accumulation_buffer) cudaFree(accumulation_buffer);
        if (d_scene) cudaFree(d_scene);
        if (cuda_texture_resource) cudaFree(cuda_texture_resource);
        if (gl_texture) glDeleteTextures(1, &gl_texture);
}

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

__global__ void render_kernel(float4 *accumulation_buffer, cudaSurfaceObject_t output_surf, RenderConfig config,
                              Camera cam, Scene *world, int sample_count, int seed) {
        // calculate pixel coordinates
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // exit if out of bounds
        if (x >= config.image_w || y >= config.image_h) return;

        // compute 1d index for accumulation buffer
        int idx = y * config.image_w + x;

        // initialize random state for this pixel
        RandState rand_state;
        random_init(&rand_state, seed, idx);

        // accumulate color
        float u = float(x + random_float(&rand_state)) / float(config.image_w - 1);
        float v = float(config.image_h - 1 - y + random_float(&rand_state)) / float(config.image_h - 1);
        Ray ray = cam.get_ray(u, v, &rand_state);

        Vec3 current_color = ray_color(ray, world, &rand_state, config.max_depth);

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
        // uchar4 pixel = make_uchar4(255, 0, 0, 255);

        // write pixel to surface, flipping y to match opengl's coordinate system
        surf2Dwrite(pixel, output_surf, x * sizeof(uchar4), config.image_h - 1 - y, cudaBoundaryModeClamp);
}

void Renderer::init(const int width, const int height, RenderConfig render_config) {
        window_w = width;
        window_h = height;
        config   = render_config;

        sample_count = 0;

        size_t buffer_size = window_w * window_h * sizeof(float4);
        CHECK_CUDA_ERROR(cudaMalloc(&accumulation_buffer, buffer_size));
        CHECK_CUDA_ERROR(cudaMemset(accumulation_buffer, 0, buffer_size));

        glGenTextures(1, &gl_texture);
        glBindTexture(GL_TEXTURE_2D, gl_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, window_w, window_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
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

void Renderer::render(const Scene &scene, const Camera &camera, bool reset_accumulation) {
        if (reset_accumulation) {
                CHECK_CUDA_ERROR(cudaMemset(accumulation_buffer, 0, window_w * window_h * sizeof(float4)));
                sample_count = 0;
        }

        // update scene on device if needed
        if (scene_needs_update) {
                CHECK_CUDA_ERROR(cudaMemcpy(d_scene, &scene, sizeof(Scene), cudaMemcpyHostToDevice));
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
        render_kernel<<<grid_size, block_size>>>(accumulation_buffer, output_surf, config, camera, d_scene,
                                                 sample_count, time(NULL));

        // check for kernel launch or execution errors
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // clean up the surface object
        CHECK_CUDA_ERROR(cudaDestroySurfaceObject(output_surf));

        // unmap the texture resource
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cuda_texture_resource, 0));
}
