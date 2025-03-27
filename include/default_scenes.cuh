#pragma once

#include "scene.cuh"
#include "sphere.cuh"
#include "vec3.cuh"

namespace default_scenes {

// final scene from the first book, but with moving spheres.
void moving_spheres(Scene &scene) {
        // ground material
        int ground_material = scene.add_material(Material(LAMBERTIAN, Vec3(0.5f, 0.5f, 0.5f)));

        // large sphere for the ground
        scene.add_sphere(Sphere(Vec3(0, -1000, 0), 1000, ground_material));

        // three main spheres
        int material1 = scene.add_material(Material(DIELECTRIC, Vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f));
        int material2 = scene.add_material(Material(LAMBERTIAN, Vec3(0.4f, 0.2f, 0.1f)));
        int material3 = scene.add_material(Material(METAL, Vec3(0.7f, 0.6f, 0.5f), 0.0f));

        scene.add_sphere(Sphere(Vec3(-4, 1, 0), 1.0f, material2));
        scene.add_sphere(Sphere(Vec3(0, 1, 0), 1.0f, material1));
        scene.add_sphere(Sphere(Vec3(4, 1, 0), 1.0f, material3));

        // small spheres
        if (!false) {
                for (int a = -11; a < 11; a++) {
                        for (int b = -11; b < 11; b++) {
                                float choose_mat = random_float();
                                Vec3 center(a + 0.9f * random_float(), 0.2f, b + 0.9f * random_float());

                                if ((center - Vec3(4, 0.2f, 0)).length() > 0.9f) {
                                        int sphere_material;

                                        if (choose_mat < 0.8f) {
                                                // diffuse
                                                Vec3 albedo(random_float() * random_float(),
                                                            random_float() * random_float(),
                                                            random_float() * random_float());
                                                sphere_material = scene.add_material(Material(LAMBERTIAN, albedo));
                                                Vec3 center2    = center + Vec3(0, random_float(0.0f, 0.5f), 0.0f);
                                                scene.add_sphere(Sphere(center, center2, 0.2f, sphere_material));
                                        } else if (choose_mat < 0.95f) {
                                                // metal
                                                Vec3 albedo(0.5f * (1 + random_float()), 0.5f * (1 + random_float()),
                                                            0.5f * (1 + random_float()));
                                                float fuzz      = 0.5f * random_float();
                                                sphere_material = scene.add_material(Material(METAL, albedo, fuzz));
                                                scene.add_sphere(Sphere(center, 0.2f, sphere_material));
                                        } else {
                                                // glass
                                                sphere_material = scene.add_material(
                                                        Material(DIELECTRIC, Vec3(1.0f, 1.0f, 1.0f), 0.0f, 1.5f));
                                                scene.add_sphere(Sphere(center, 0.2f, sphere_material));
                                        }
                                }
                        }
                }
        }
}

}  // namespace default_scenes
