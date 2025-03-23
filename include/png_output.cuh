#pragma once

#include <png.h>

// save a framebuffer to a png file
__host__ bool save_png(const char *filename, unsigned int *framebuffer, int width, int height);
