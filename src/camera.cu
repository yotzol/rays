#include "camera.cuh"

__host__ Camera::Camera(float vfov, float aspect_ratio) {
        this->vfov         = vfov;
        this->aspect_ratio = aspect_ratio;
        this->aperture     = 0.0f;
        this->focus_dist   = 1.0f;

        // default position
        origin = Vec3(0, 0, 0);

        yaw   = -90.0f;  // looking along negative z-axis
        pitch = 0.0f;
        roll  = 0.0f;

        // default movement settings
        movement_speed    = camera::DEFAULT_MOVEMENT_SPEED;
        mouse_sensitivity = camera::DEFAULT_MOUSE_SENSITIVITY;
        tilt_speed        = camera::DEFAULT_TILT_SPEED;

        lens_radius = aperture / 2.0f;

        // initialize camera vectors
        update_camera();
}

__host__ Camera::Camera(Vec3 lookfrom, Vec3 lookat, float vfov, float aspect_ratio, float aperture,
                        float focus_dist) {
        this->vfov         = vfov;
        this->aspect_ratio = aspect_ratio;
        this->aperture     = aperture;
        this->focus_dist   = focus_dist;

        // position
        origin = lookfrom;

        // calculate initial yaw and pitch from lookfrom and lookat
        Vec3 direction = normalize(lookat - lookfrom);
        yaw            = degrees(atan2(direction.z, direction.x));
        pitch          = degrees(asin(direction.y));
        roll           = 0.0f;

        // default movement settings
        movement_speed    = camera::DEFAULT_MOVEMENT_SPEED;
        mouse_sensitivity = camera::DEFAULT_MOUSE_SENSITIVITY;
        tilt_speed        = camera::DEFAULT_TILT_SPEED;

        lens_radius = aperture / 2.0f;

        // initialize camera vectors
        update_camera();
}

__device__ Ray Camera::get_ray(float s, float t, RandState *rand_state) const {
        Vec3 rd = Vec3(0, 0, 0);

        // defocus blur if lens_radius > 0
        if (lens_radius > 0) {
                rd = random_in_unit_disk(rand_state) * lens_radius;
                rd = u * rd.x + v * rd.y;
        }

        float time = random_float(rand_state);

        return Ray(origin + rd, lower_left_corner + horizontal * s + vertical * t - origin - rd, time);
}


const Vec3 vec_up          = Vec3(0.0f, 1.0f, 0.0f);
const Vec3 cancel_vertical = Vec3(1.0f, 0.0f, 1.0f);

__host__ void Camera::move(CameraMovement direction) {
        float velocity = movement_speed;

        switch (direction) {
                case CameraMovement::FORWARD : origin -= w * cancel_vertical * velocity; break;
                case CameraMovement::BACKWARD: origin += w * cancel_vertical * velocity; break;
                case CameraMovement::LEFT    : origin -= u * cancel_vertical * velocity; break;
                case CameraMovement::RIGHT   : origin += u * cancel_vertical * velocity; break;
                case CameraMovement::UP      : origin += vec_up * velocity; break;
                case CameraMovement::DOWN    : origin -= vec_up * velocity; break;
        }

        // update camera parameters after movement
        update_camera();
}

__host__ void Camera::process_mouse_movement(float x_offset, float y_offset, bool constrain_pitch) {
        x_offset *= mouse_sensitivity;
        y_offset *= mouse_sensitivity;

        yaw += x_offset;
        pitch += y_offset;

        // prevent gimbal lock by constraining pitch
        if (constrain_pitch) {
                if (pitch >  89.0f) pitch =  89.0f;
                if (pitch < -89.0f) pitch = -89.0f;
        }

        // update camera orientation
        update_camera();
}

// handle qe keys for camera tilt (roll)
__host__ void Camera::tilt(bool clockwise) {
        roll += clockwise ? tilt_speed : -tilt_speed;

        // update camera orientation
        update_camera();
}

// update camera vectors based on current euler angles
__host__ void Camera::update_camera() {
        // calculate camera direction vector from euler angles
        float yaw_rad   = radians(yaw);
        float pitch_rad = radians(pitch);
        float roll_rad  = radians(roll);

        // calculate front vector (opposite of w)
        Vec3 front;
        front.x = cos(yaw_rad) * cos(pitch_rad);
        front.y = sin(pitch_rad);
        front.z = sin(yaw_rad) * cos(pitch_rad);
        front   = normalize(front);

        // w points opposite to front direction
        w = -front;

        // calculate right vector (u)
        Vec3 world_up = Vec3(0, 1, 0);
        u             = normalize(cross(world_up, w));  // right vector

        // calculate up vector (v), incorporating roll if needed
        v = normalize(cross(w, u));

        // apply roll rotation if needed
        if (roll_rad != 0.0f) {
                float cos_roll = cos(roll_rad);
                float sin_roll = sin(roll_rad);

                Vec3 new_u = u * cos_roll - v * sin_roll;
                Vec3 new_v = u * sin_roll + v * cos_roll;

                u = new_u;
                v = new_v;
        }

        // recalculate camera parameters
        float theta           = vfov * M_PI / 180.0f;
        float h               = tan(theta / 2);
        float viewport_height = 2.0f * h;
        float viewport_width  = aspect_ratio * viewport_height;

        horizontal        = u * viewport_width * focus_dist;
        vertical          = v * viewport_height * focus_dist;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - w * focus_dist;
}
