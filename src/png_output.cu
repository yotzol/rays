#include "png_output.cuh"

#include <stdio.h>
#include <stdlib.h>

__host__ bool save_png(const char *filename, unsigned int *framebuffer, int width, int height) {
        // open file for writing (binary mode)
        FILE *fp = fopen(filename, "wb");
        if (!fp) {
                fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
                return false;
        }

        // initialize write structure
        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (!png_ptr) {
                fprintf(stderr, "Error: Could not create PNG write struct\n");
                fclose(fp);
                return false;
        }

        // initialize info structure
        png_infop info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr) {
                fprintf(stderr, "Error: Could not create PNG info struct\n");
                png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
                fclose(fp);
                return false;
        }

        // setup error handling
        if (setjmp(png_jmpbuf(png_ptr))) {
                fprintf(stderr, "Error during PNG file creation\n");
                png_destroy_write_struct(&png_ptr, &info_ptr);
                fclose(fp);
                return false;
        }

        // set output
        png_init_io(png_ptr, fp);

        // write header (8-bit rgb color)
        png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png_ptr, info_ptr);

        // allocate memory for row pointers
        png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
        if (!row_pointers) {
                fprintf(stderr, "Error: Could not allocate memory for PNG row pointers\n");
                png_destroy_write_struct(&png_ptr, &info_ptr);
                fclose(fp);
                return false;
        }

        // allocate memory for data
        for (int y = 0; y < height; y++) {
                row_pointers[y] = (png_byte *)malloc(width * 3);
                if (!row_pointers[y]) {
                        fprintf(stderr, "Error: Could not allocate memory for PNG row %d\n", y);
                        for (int j = 0; j < y; j++) {
                                free(row_pointers[j]);
                        }
                        free(row_pointers);
                        png_destroy_write_struct(&png_ptr, &info_ptr);
                        fclose(fp);
                        return false;
                }
        }

        // fill row data from framebuffer (rgba -> rgb)
        for (int y = 0; y < height; y++) {
                png_bytep row = row_pointers[y];
                for (int x = 0; x < width; x++) {
                        unsigned int pixel = framebuffer[y * width + x];

                        // extract rgb components (rgba format with a in most significant byte)
                        unsigned char r = pixel & 0xFF;
                        unsigned char g = (pixel >> 8) & 0xFF;
                        unsigned char b = (pixel >> 16) & 0xFF;

                        int idx      = x * 3;
                        row[idx + 0] = r;
                        row[idx + 1] = g;
                        row[idx + 2] = b;
                }
        }

        // write image data
        png_write_image(png_ptr, row_pointers);
        png_write_end(png_ptr, NULL);

        // free memory
        for (int y = 0; y < height; y++) {
                free(row_pointers[y]);
        }
        free(row_pointers);

        // clean up
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);

        return true;
}
