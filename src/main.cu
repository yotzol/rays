#include "camera.cuh"
#include "material.cuh"
#include "png_output.cuh"
#include "render.cuh"
#include "scene.cuh"
#include "sphere.cuh"
#include "vec3.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// final scene from the first book
void final_scene(Scene &world) {
        // ground material
        int ground_material = world.add_material(Material(LAMBERTIAN, Vec3(0.5f, 0.5f, 0.5f)));

        // large sphere for the ground
        world.add_sphere(Sphere(Vec3(0, -1000, 0), 1000, ground_material));

        // three main spheres
        int material1 = world.add_material(Material(DIELECTRIC, Vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f));
        int material2 = world.add_material(Material(LAMBERTIAN, Vec3(0.4f, 0.2f, 0.1f)));
        int material3 = world.add_material(Material(METAL, Vec3(0.7f, 0.6f, 0.5f), 0.0f));

        world.add_sphere(Sphere(Vec3(-4, 1, 0), 1.0f, material2));
        world.add_sphere(Sphere(Vec3(0, 1, 0), 1.0f, material1));
        world.add_sphere(Sphere(Vec3(4, 1, 0), 1.0f, material3));

        // small spheres
        for (int a = -11; a < 11; a++) {
                for (int b = -11; b < 11; b++) {
                        float choose_mat = (float)rand() / RAND_MAX;
                        Vec3 center(a + 0.9f * ((float)rand() / RAND_MAX), 0.2f, b + 0.9f * ((float)rand() / RAND_MAX));

                        if ((center - Vec3(4, 0.2f, 0)).length() > 0.9f) {
                                int sphere_material;

                                if (choose_mat < 0.8f) {
                                        // diffuse
                                        Vec3 albedo(((float)rand() / RAND_MAX) * ((float)rand() / RAND_MAX),
                                                    ((float)rand() / RAND_MAX) * ((float)rand() / RAND_MAX),
                                                    ((float)rand() / RAND_MAX) * ((float)rand() / RAND_MAX));
                                        sphere_material = world.add_material(Material(LAMBERTIAN, albedo));
                                } else if (choose_mat < 0.95f) {
                                        // metal
                                        Vec3 albedo(0.5f * (1 + ((float)rand() / RAND_MAX)),
                                                    0.5f * (1 + ((float)rand() / RAND_MAX)),
                                                    0.5f * (1 + ((float)rand() / RAND_MAX)));
                                        float fuzz      = 0.5f * ((float)rand() / RAND_MAX);
                                        sphere_material = world.add_material(Material(METAL, albedo, fuzz));
                                } else {
                                        // glass
                                        sphere_material = world.add_material(
                                                Material(DIELECTRIC, Vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f));
                                }

                                world.add_sphere(Sphere(center, 0.2f, sphere_material));
                        }
                }
        }
}

int main() {
        // seed the random number generator
        srand(time(NULL));

        const char output_path[] = "output.png";

        // render settings
        int image_width       = 1920;
        int image_height      = 1080;
        int samples_per_pixel = 500;
        int max_depth         = 50;

        // camera settings
        Camera camera(Vec3(13, 2, 3),                            // lookfrom
                      Vec3(0, 0, 0),                             // lookat
                      Vec3(0, 1, 0),                             // up vector
                      20.0f,                                     // vertical FOV
                      float(image_width) / float(image_height),  // aspect ratio
                      0.1f,                                      // aperture
                      10.0f                                      // focus distance
        );

        // create a test scene first
        Scene world;
        final_scene(world);

        printf("Created scene with %d spheres and %d materials\n", world.num_spheres, world.num_materials);

        // allocate memory for the framebuffer
        unsigned int *framebuffer = new unsigned int[image_width * image_height];

        // start timer
        clock_t start = clock();

        // render the scene
        printf("Rendering scene with:\n\t%d x %d resolution\n\t%d samples per pixel\n\t%d max depth\n", image_width,
               image_height, samples_per_pixel, max_depth);

        render_scene(framebuffer, image_width, image_height, samples_per_pixel, max_depth, camera, world);

        // end timer and print render time
        clock_t end         = clock();
        double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
        printf("Rendering completed in %.2f seconds.\n", elapsed_time);

        if (save_png(output_path, framebuffer, image_width, image_height)) {
                printf("Image saved successfully.\n");
        } else {
                printf("Failed to save image.\n");
        }

        // free memory
        delete[] framebuffer;

        return 0;
}
