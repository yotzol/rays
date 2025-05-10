#pragma once

#include "material.hpp"
#include "scene.hpp"
#include "object.hpp"
#include "texture.hpp"
#include "vec3.hpp"
#include "camera.hpp"
#include "utils.hpp"

namespace default_scenes {

// Final scene from the first book.
inline void moving_spheres(Scene &scene, Camera &camera) {
        // Ground material.
        int ground_material = scene.add_material(
                lambertian_new(texture_checker(Vec3(0.2f, 0.3f, 0.1f), Vec3(0.9f, 0.9f, 0.9f), 0.32)));

        // Large sphere for the ground.
        scene.add_object(sphere_new(Vec3(0, -1000, 0), 1000, ground_material));

        // Three main spheres.
        int material1 = scene.add_material(dielectric_new(1.5f));
        int material2 = scene.add_material(lambertian_new(texture_solid(Vec3(0.4f, 0.2f, 0.1f))));
        int material3 = scene.add_material(metal_new(Vec3(0.7f, 0.6f, 0.5f), 0.0f));

        scene.add_object(sphere_new(Vec3(-4, 1, 0), 1.0f, material2));
        scene.add_object(sphere_new(Vec3( 0, 1, 0), 1.0f, material1));
        scene.add_object(sphere_new(Vec3( 4, 1, 0), 1.0f, material3));

        // Small spheres.
        if (!false) {
                for (int a = -11; a < 11; a++) {
                        for (int b = -11; b < 11; b++) {
                                float choose_mat = randf();
                                Vec3 center(a + 0.9f * randf(), 0.2f, b + 0.9f * randf());

                                if ((center - Vec3(4, 0.2f, 0)).length() > 0.9f) {
                                        int sphere_material;

                                        if (choose_mat < 0.8f) {
                                                // Diffuse.
                                                Vec3 albedo(randf() * randf(),
                                                            randf() * randf(),
                                                            randf() * randf());
                                                sphere_material =
                                                        scene.add_material(lambertian_new(texture_solid(albedo)));
                                                Vec3 center2 = center + Vec3(0, randf(0.0f, 0.5f), 0.0f);
                                                scene.add_object(sphere_moving(center, center2, 0.2f, sphere_material));
                                        } else if (choose_mat < 0.95f) {
                                                // Metal.
                                                Vec3 albedo(0.5f * (1 + randf()), 0.5f * (1 + randf()),
                                                            0.5f * (1 + randf()));
                                                float fuzz      = 0.5f * randf();
                                                sphere_material = scene.add_material(metal_new(albedo, fuzz));
                                                scene.add_object(sphere_new(center, 0.2f, sphere_material));
                                        } else {
                                                // Glass.
                                                sphere_material = scene.add_material(dielectric_new(1.5f));
                                                scene.add_object(sphere_new(center, 0.2f, sphere_material));
                                        }
                                }
                        }
                }
        }

        camera = Camera(Vec3(13, 1, 0),       // Lookfrom.
                        Vec3(0, 1, 0),        // Lookat.
                        20.0f,                // Vertical fov.
                        camera.aspect_ratio,  // Aspect ratio.
                        0.0f,                 // Aperture.
                        10.0f                 // Focus distance.
        );
}

inline void quads(Scene &scene, Camera &camera) {
        int left_red     = scene.add_material(lambertian_new(texture_solid(Vec3(1.0f, 0.2f, 0.2f))));
        int back_green   = scene.add_material(lambertian_new(texture_solid(Vec3(0.2f, 1.0f, 0.2f))));
        int right_blue   = scene.add_material(lambertian_new(texture_solid(Vec3(0.2f, 0.2f, 1.0f))));
        int upper_orange = scene.add_material(lambertian_new(texture_solid(Vec3(1.0f, 0.5f, 0.0f))));
        int lower_teal   = scene.add_material(lambertian_new(texture_solid(Vec3(0.2f, 0.8f, 0.8f))));

        scene.add_object(quad_new(Vec3(-3, -2, 5), Vec3(0, 0, -4), Vec3(0, 4,  0), left_red));
        scene.add_object(quad_new(Vec3(-2, -2, 0), Vec3(4, 0,  0), Vec3(0, 4,  0), back_green));
        scene.add_object(quad_new(Vec3( 3, -2, 1), Vec3(0, 0,  4), Vec3(0, 4,  0), right_blue));
        scene.add_object(quad_new(Vec3(-2,  3, 1), Vec3(4, 0,  0), Vec3(0, 0,  4), upper_orange));
        scene.add_object(quad_new(Vec3(-2, -3, 5), Vec3(4, 0,  0), Vec3(0, 0, -4), lower_teal));

        camera = Camera(Vec3(0, 0, 9),        // Lookfrom.
                        Vec3(0, 0, 0),        // Lookat.
                        20.0f,                // Vertical fov.
                        camera.aspect_ratio,  // Aspect ratio.
                        0.0f,                 // Aperture.
                        10.0f                 // Focus distance.
        );
}

inline void simple_light(Scene &scene, Camera &camera) {
        int ground_material = scene.add_material(lambertian_new(texture_solid(Vec3(1,1,1))));
        int sphere_material = scene.add_material(lambertian_new(texture_solid(Vec3(0.8f, 0.2f, 1.0f))));

        scene.add_object(sphere_new(Vec3(0, -1000, 0), 1000.0f, ground_material));
        scene.add_object(sphere_new(Vec3(0,     2, 0),    2.0f, sphere_material));

        int light_material = scene.add_material(diffuse_light_new(texture_solid(Vec3(4.0f, 4.0f, 4.0f))));
        scene.add_object(quad_new(Vec3(3, 1, -2), Vec3(2, 0, 0), Vec3(0, 2, 0), light_material));
        scene.add_object(sphere_new(Vec3(0, 7, 0), 2.0f, light_material));

        scene.no_environment   = true;
        scene.background_color = Vec3(0, 0, 0);

        camera = Camera(Vec3(26, 3, 6),  // Lookfrom.
                        Vec3(0, 2, 0),   // Lookat.
                        20.0f,           // Vertical fov.
                        camera.aspect_ratio,
                        0.0f,            // Aperture.
                        10.0f            // Focus distance.
        );
}

inline void cornell_box(Scene &scene, Camera &camera) {
        int red   = scene.add_material(lambertian_new(texture_solid(Vec3(0.65f, 0.05f, 0.05f))));
        int white = scene.add_material(lambertian_new(texture_solid(Vec3(0.73f, 0.73f, 0.73f))));
        int green = scene.add_material(lambertian_new(texture_solid(Vec3(0.12f, 0.45f, 0.15f))));
        int light = scene.add_material(diffuse_light_new(texture_solid(Vec3(15.0f, 15.0f, 15.0f))));

        scene.add_object(quad_new(Vec3(555,   0,   0), Vec3(   0, 555, 0), Vec3(0,   0,  555), green)); // Left.
        scene.add_object(quad_new(Vec3(  0,   0,   0), Vec3(   0, 555, 0), Vec3(0,   0,  555),   red)); // Right.
        scene.add_object(quad_new(Vec3(  0,   0,   0), Vec3( 555,   0, 0), Vec3(0,   0,  555), white)); // Bottom.
        scene.add_object(quad_new(Vec3(555, 555, 555), Vec3(-555,   0, 0), Vec3(0,   0, -555), white)); // Top.
        scene.add_object(quad_new(Vec3(  0,   0, 555), Vec3( 555,   0, 0), Vec3(0, 555,    0), white)); // Back.
        scene.add_object(quad_new(Vec3(343, 554, 332), Vec3(-130,   0, 0), Vec3(0,   0, -105), light)); // Light.

        scene.add_box(Vec3(0, 0, 0), Vec3(165, 330, 165), white,  15, Vec3(265, 0, 295));
        scene.add_box(Vec3(0, 0, 0), Vec3(165, 165, 165), white, 360-18, Vec3(130, 0,  65));

        scene.no_environment   = true;
        scene.background_color = Vec3(0, 0, 0);

        camera = Camera(Vec3(278, 278, -800), Vec3(278, 278, 0), 40.0f, 1, 0.0f, 1.0f);
}

inline void book2_final_scene(Scene &scene, Camera &camera) {
        // Ground boxes.
        int ground_material = scene.add_material(lambertian_new(texture_solid(Vec3(0.48f, 0.83f, 0.53f))));
        int boxes_per_side  = 20;
        for (int i = 0; i < boxes_per_side; i++) {
                for (int j = 0; j < boxes_per_side; j++) {
                        float w  = 100.0f;
                        float x0 = -1000.0f + i * w;
                        float z0 = -1000.0f + j * w;
                        float y0 = 0.0f;
                        float x1 = x0 + w;
                        float y1 = randf(1.0f, 101.0f);
                        float z1 = z0 + w;
                        scene.add_box(Vec3(x0, y0, z0), Vec3(x1, y1, z1), ground_material, 0.0f, Vec3(0, 0, 0));
                }
        }

        // Light.
        int light_material = scene.add_material(diffuse_light_new(texture_solid(Vec3(7.0f, 7.0f, 7.0f))));
        scene.add_object(quad_new(Vec3(123, 554, 147), Vec3(300, 0, 0), Vec3(0, 0, 265), light_material));

        // Moving sphere.
        int sphere_material = scene.add_material(lambertian_new(texture_solid(Vec3(0.7f, 0.3f, 0.1f))));
        Vec3 center1(400, 400, 200);
        Vec3 center2 = center1 + Vec3(30, 0, 0);
        scene.add_object(sphere_moving(center1, center2, 50.0f, sphere_material));

        // Dielectric sphere.
        int dielectric_material = scene.add_material(dielectric_new(1.5f));
        scene.add_object(sphere_new(Vec3(260, 150, 45), 50.0f, dielectric_material));

        // Metal sphere.
        int metal_material = scene.add_material(metal_new(Vec3(0.8f, 0.8f, 0.9f), 1.0f));
        scene.add_object(sphere_new(Vec3(0, 150, 145), 50.0f, metal_material));

        // Dielectric sphere with lambertian interior (replacing constant medium).
        int inner_material = scene.add_material(lambertian_new(texture_solid(Vec3(0.2f, 0.4f, 0.9f))));
        scene.add_object(sphere_new(Vec3(360, 150, 145), 70.0f, dielectric_material));
        scene.add_object(sphere_new(Vec3(360, 150, 145), 69.0f, inner_material));

        // Earth sphere.
        int earth_material = scene.add_material(lambertian_new(texture_image("earth2.png")));
        scene.add_object(sphere_new(Vec3(400, 200, 400), 100.0f, earth_material));

        // Metal sphere (replacing noise texture sphere).
        int noise_material = scene.add_material(metal_new(Vec3(0.7f, 0.7f, 0.7f), 0.2f));
        scene.add_object(sphere_new(Vec3(220, 280, 300), 80.0f, noise_material));

        // Small white spheres.
        int white_material = scene.add_material(lambertian_new(texture_solid(Vec3(0.73f, 0.73f, 0.73f))));
        int ns             = 1000;
        for (int j = 0; j < ns; j++) {
                Vec3 center = Vec3(randf(0.0f, 165.0f), randf(0.0f, 165.0f), randf(0.0f, 165.0f));
                scene.add_object(sphere_new(center, 10.0f, white_material), 15, Vec3(-100, 270, 395), Vec3(0));
        }

        // Scene settings.
        scene.no_environment   = true;
        scene.background_color = Vec3(0, 0, 0);

        // Camera.
        camera = Camera(Vec3(478, 278, -600),  // Lookfrom.
                        Vec3(278, 278, 0),     // Lookat.
                        40.0f,                 // Vertical fov.
                        camera.aspect_ratio,   // Aspect ratio.
                        0.0f,                  // Aperture.
                        10.0f                  // Focus distance.
        );
}

}  // namespace default_scenes
