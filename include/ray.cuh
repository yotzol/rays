#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

class __align__(16) Ray {
       public:
        Vec3 origin;
        Vec3 direction;
        float time;

        __host__ __device__ Ray() : time(0) {}
        __host__ __device__ Ray(const Vec3 &origin, const Vec3 &direction, const float t = 0)
            : origin(origin), direction(direction), time(t) {}

        __device__ __forceinline__ Vec3 at(float t) const {
                return origin + direction * t;
        }
};

#endif  // RAY_CUH
