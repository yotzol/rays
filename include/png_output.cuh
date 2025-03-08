#ifndef PNG_OUTPUT_CUH
#define PNG_OUTPUT_CUH

#include <png.h>

// save a framebuffer to a PNG file
bool save_png(const char *filename, unsigned int *framebuffer, int width, int height);

#endif  // PNG_OUTPUT_CUH
