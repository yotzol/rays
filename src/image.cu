#include "image.cuh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

__host__ bool save_png(const char *filename, unsigned int *framebuffer, int width, int height) {
        return stbi_write_png(filename, width, height, 4, framebuffer, 4 * width);
}
