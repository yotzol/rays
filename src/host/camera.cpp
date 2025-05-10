#include "camera.hpp"

__host__ Camera::Camera(float vfov, float aspect_ratio) {
        this->vfov         = vfov;
        this->aspect_ratio = aspect_ratio;
        this->aperture     = 0.0f;
        this->focus_dist   = 1.0f;

        // Default position.
        origin = Vec3(0, 0, 0);

        yaw   = -90.0f;  // Looking along negative z-axis.
        pitch = 0.0f;
        roll  = 0.0f;

        world_up = Vec3(0, 1, 0);

        // Default movement settings.
        movement_speed    = camera::DEFAULT_MOVEMENT_SPEED;
        mouse_sensitivity = camera::DEFAULT_MOUSE_SENSITIVITY;
        tilt_speed        = camera::DEFAULT_TILT_SPEED;

        lens_radius = aperture / 2.0f;

        // Compute camera vectors.
        update_camera();
}

__host__ Camera::Camera(Vec3 lookfrom, Vec3 lookat, float vfov, float aspect_ratio, float aperture, float focus_dist) {
        this->vfov         = vfov;
        this->aspect_ratio = aspect_ratio;
        this->aperture     = aperture;
        this->focus_dist   = focus_dist;

        // Position.
        origin = lookfrom;

        // Calculate initial yaw and pitch from lookfrom and lookat.
        Vec3 direction = normalize(lookat - lookfrom);
        yaw            = to_degrees(atan2(direction.z, direction.x));
        pitch          = to_degrees(asin(direction.y));
        roll           = 0.0f;

        world_up = Vec3(0, 1, 0);

        // Default movement settings.
        movement_speed    = camera::DEFAULT_MOVEMENT_SPEED;
        mouse_sensitivity = camera::DEFAULT_MOUSE_SENSITIVITY;
        tilt_speed        = camera::DEFAULT_TILT_SPEED;

        lens_radius = aperture / 2.0f;

        // Compute camera vectors.
        update_camera();
}

__host__ void Camera::move(CameraMovement direction) {
        float velocity = movement_speed;

        const Vec3 cancel_vertical = Vec3(1.0f, 0.0f, 1.0f);

        switch (direction) {
                case CameraMovement::FORWARD : origin -= w * cancel_vertical * velocity; break;
                case CameraMovement::BACKWARD: origin += w * cancel_vertical * velocity; break;
                case CameraMovement::LEFT    : origin -= u * cancel_vertical * velocity; break;
                case CameraMovement::RIGHT   : origin += u * cancel_vertical * velocity; break;
                case CameraMovement::UP      : origin += world_up * velocity; break;
                case CameraMovement::DOWN    : origin -= world_up * velocity; break;
        }

        // Update camera parameters after movement.
        update_camera();
}

__host__ void Camera::process_mouse_movement(float x_offset, float y_offset, bool constrain_pitch) {
        x_offset *= mouse_sensitivity;
        y_offset *= mouse_sensitivity;

        yaw += x_offset;
        pitch += y_offset;

        // Prevent gimbal lock by constraining pitch.
        if (constrain_pitch) {
                if (pitch > 89.0f) pitch = 89.0f;
                if (pitch < -89.0f) pitch = -89.0f;
        }
        update_camera();
}

// Handle QE keys for camera tilt (roll).
__host__ void Camera::tilt(bool clockwise) {
        roll += clockwise ? tilt_speed : -tilt_speed;
        update_camera();
}

// Update camera vectors based on current Euler angles.
__host__ void Camera::update_camera() {
        // Calculate camera direction vector from euler angles.
        float yaw_rad   = to_radians(yaw);
        float pitch_rad = to_radians(pitch);
        float roll_rad  = to_radians(roll);

        // Calculate front vector.
        Vec3 front;
        front.x = cos(yaw_rad) * cos(pitch_rad);
        front.y = sin(pitch_rad);
        front.z = sin(yaw_rad) * cos(pitch_rad);
        front   = normalize(front);

        // Calculate camera vectors.
        w = -front;                         // W points ooposite to front direction.
        u = normalize(cross(world_up, w));  // Right vector.
        v = normalize(cross(w, u));         // Up Vector.

        // Apply roll rotation if needed.
        if (roll_rad != 0.0f) {
                float cos_roll = cos(roll_rad);
                float sin_roll = sin(roll_rad);

                Vec3 new_u = u * cos_roll - v * sin_roll;
                Vec3 new_v = u * sin_roll + v * cos_roll;

                u = new_u;
                v = new_v;
        }

        // Recalculate camera parameters.
        float theta           = to_radians(vfov);
        float h               = tan(theta / 2);
        float viewport_height = 2.0f * h;
        float viewport_width  = aspect_ratio * viewport_height;

        horizontal        = u * viewport_width * focus_dist;
        vertical          = v * viewport_height * focus_dist;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - w * focus_dist;

        moved = true;
}
