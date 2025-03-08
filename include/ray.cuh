#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

class __align__(16) Ray {
       public:
        Vec3 origin;
        Vec3 direction;

        __host__ __device__ Ray() {}
        __host__ __device__ Ray(const Vec3 &origin, const Vec3 &direction) : origin(origin), direction(direction) {}

        __device__ __forceinline__ Vec3 at(float t) const {
                return origin + direction * t;
        }
};

#endif  // RAY_CUH
