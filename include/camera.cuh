#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "ray.cuh"
#include "utils.cuh"

class __align__(16) Camera {
       public:
        Vec3 origin;
        Vec3 lower_left_corner;
        Vec3 horizontal;
        Vec3 vertical;
        Vec3 u, v, w;       // camera basis vectors
        float lens_radius;  // for depth of field

        // simple camera constructor
        __host__ Camera() {
                origin            = Vec3(0, 0, 0);
                lower_left_corner = Vec3(-2, -1, -1);
                horizontal        = Vec3(4, 0, 0);
                vertical          = Vec3(0, 2, 0);
                lens_radius       = 0;
        }

        // advanced camera constructor
        __host__ __device__ Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect_ratio, float aperture,
                                   float focus_dist) {
                float theta           = vfov * M_PI / 180.0f;
                float h               = tanf(theta / 2);
                float viewport_height = 2.0f * h;
                float viewport_width  = aspect_ratio * viewport_height;

                w = normalize(lookfrom - lookat);
                u = normalize(cross(vup, w));
                v = cross(w, u);

                origin            = lookfrom;
                horizontal        = u * viewport_width * focus_dist;
                vertical          = v * viewport_height * focus_dist;
                lower_left_corner = origin - horizontal / 2 - vertical / 2 - w * focus_dist;
                lens_radius       = aperture / 2;
        }

        // generate a ray from the camera through a given pixel
        __device__ Ray get_ray(float s, float t, RandState *rand_state) const {
                Vec3 rd = Vec3(0, 0, 0);

                // defocus blur if lens_radius > 0
                if (lens_radius > 0) {
                        rd = random_in_unit_disk(rand_state) * lens_radius;
                        rd = u * rd.x + v * rd.y;
                }

                return Ray(origin + rd, lower_left_corner + horizontal * s + vertical * t - origin - rd);
        }
};

#endif  // CAMERA_CUH
