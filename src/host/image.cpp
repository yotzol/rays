#include "image.hpp"

#include "utils.hpp"

#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

cudaTextureObject_t load_texture(const char filename[]) {
        std::string path   = "../assets/textures/" + std::string(filename);
        const char *c_path = path.c_str();
        return load_asset(c_path);
}

cudaTextureObject_t load_envmap(const char filename[]) {
        std::string path   = "../assets/envmaps/" + std::string(filename);
        const char *c_path = path.c_str();
        return load_asset(c_path);
}

// Load file from assets/ directory into CUDA texture.
cudaTextureObject_t load_asset(const char path[]) {
        cudaTextureObject_t tex_obj;
        int w, h, channels;
        unsigned char *data = stbi_load(path, &w, &h, &channels, 4);

        if (!data) {
                fprintf(stderr, "Failed to load image: %s\n", path);
                return 0;
        }

        // Print image info.
        printf("Loaded asset\"%s\":\n", path);
        printf("\tW: %d\n\tH: %d\n", w, h);
        fflush(stdout);

        // RGBA channel format.
        const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

        // Copy image data to device.
        cudaArray *d_array;
        CHECK_CUDA_ERROR(cudaMallocArray(&d_array, &channel_desc, (size_t)w, (size_t)h));
        const size_t src_pitch = (unsigned long)w * 4 * sizeof(unsigned char);
        CHECK_CUDA_ERROR(cudaMemcpy2DToArray(d_array, 0, 0, data, src_pitch, src_pitch, (size_t)h, cudaMemcpyHostToDevice));

        // Resource descriptor.
        cudaResourceDesc res_desc{};
        res_desc.resType         = cudaResourceTypeArray;
        res_desc.res.array.array = d_array;

        // Texture descriptor.
        cudaTextureDesc tex_desc{};
        tex_desc.addressMode[0]   = cudaAddressModeClamp;         // Clamp horizontally.
        tex_desc.addressMode[1]   = cudaAddressModeClamp;         // Clamp vertically.
        tex_desc.filterMode       = cudaFilterModeLinear;         // Linear interpolation.
        tex_desc.readMode         = cudaReadModeNormalizedFloat;  // Return normalized floats [0,1].
        tex_desc.normalizedCoords = 1;                            // Use [0,1] coordinates.

        // Create texture object.
        CHECK_CUDA_ERROR(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

        // Free host file memory.
        stbi_image_free(data);

        return tex_obj;
}

__host__ bool save_png(const char *filename, unsigned int *framebuffer, int width, int height) {
        return stbi_write_png(filename, width, height, 4, framebuffer, 4 * width);
}
