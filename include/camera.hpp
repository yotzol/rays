#pragma once

#include "vec3.hpp"

enum class CameraMovement {
        FORWARD,
        BACKWARD,
        LEFT,
        RIGHT,
        UP,
        DOWN,
};

namespace camera {

const float DEFAULT_MOVEMENT_SPEED    = 0.4f;
const float DEFAULT_MOUSE_SENSITIVITY = 0.4f;
const float DEFAULT_TILT_SPEED        = 0.005f;

}  // namespace camera

class __align__(16) Camera {
       public:
        Vec3 origin;
        Vec3 lower_left_corner;
        Vec3 horizontal;
        Vec3 vertical;
        Vec3 u, v, w;  // Camera basis vectors.
        float lens_radius;

        Vec3 world_up;

        // Camera view parameters.
        float vfov;  // Vertical field of view in degrees.
        float aspect_ratio;
        float aperture;
        float focus_dist;

        // Camera orientation (euler angles).
        float yaw;    // Rotation around y axis (left/right).
        float pitch;  // Rotation around x axis (up/down).
        float roll;   // Rotation around z axis (tilt).

        // movement parameters
        float movement_speed;
        float mouse_sensitivity;
        float tilt_speed;
        bool moved;

        // Parameterized default constructor.
        __host__ Camera(float vfov = 90.0f, float aspect_ratio = 2.0f);

        // Advanced camera constructor.
        __host__ Camera(Vec3 lookfrom, Vec3 lookat, float vfov, float aspect_ratio, float aperture = 0.0f,
                        float focus_dist = 1.0f);

#ifdef __CUDACC__
        // Generate a ray from the camera through a given pixel.
        __device__ Ray get_ray(float s, float t, curandState *rand_state) const {
                Vec3 rd = Vec3(0, 0, 0);

                // Defocus blur if lens_radius > 0.
                if (lens_radius > 0) {
                        rd = random_in_unit_disk(rand_state) * lens_radius;
                        rd = u * rd.x + v * rd.y;
                }

                float time = randf(rand_state);

                return Ray(origin + rd, normalize(lower_left_corner + horizontal * s + vertical * t - origin - rd));
        }
#endif

        __host__ void move(CameraMovement direction);

        __host__ void process_mouse_movement(float x_offset, float y_offset, bool constrain_pitch = true);

        __host__ void tilt(bool clockwise);

        __host__ void update_camera();
};
