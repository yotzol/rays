#pragma once

#include "ray.cuh"
#include "utils.cuh"

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
const float DEFAULT_MOUSE_SENSITIVITY = 0.01f;
const float DEFAULT_TILT_SPEED        = 0.005f;

}  // namespace camera

class __align__(16) Camera {
       public:
        Vec3 origin;
        Vec3 lower_left_corner;
        Vec3 horizontal;
        Vec3 vertical;
        Vec3 u, v, w;       // camera basis vectors
        float lens_radius;  // for depth of field

        // camera view parameters
        float vfov;  // vertical field of view in degrees
        float aspect_ratio;
        float aperture;
        float focus_dist;

        // camera orientation (euler angles)
        float yaw;    // rotation around y axis (left/right)
        float pitch;  // rotation around x axis (up/down)
        float roll;   // rotation around z axis (tilt)

        // movement parameters
        float movement_speed;
        float mouse_sensitivity;
        float tilt_speed;

        // parameterized default constructor
        __host__ Camera(float vfov = 90.0f, float aspect_ratio = 2.0f);

        // advanced camera constructor
        __host__ Camera(Vec3 lookfrom, Vec3 lookat, float vfov, float aspect_ratio,
                                   float aperture = 0.0f, float focus_dist = 1.0f);

        // generate a ray from the camera through a given pixel
        __device__ Ray get_ray(float s, float t, RandState *rand_state) const;

        __host__ void move(CameraMovement direction);

        __host__ void process_mouse_movement(float x_offset, float y_offset, bool constrain_pitch = true);

        __host__ void tilt(bool clockwise);

        __host__ void update_camera();
};
