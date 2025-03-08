#ifndef VEC3_CUH
#define VEC3_CUH

class __align__(16) Vec3 {
       public:
        float x, y, z;
        float pad;

        // constructors
        __host__ __device__ Vec3() : x(0), y(0), z(0), pad(0) {}
        __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z), pad(0) {}

        // operator overloading
        __host__ __device__ __forceinline__ Vec3 operator+(const Vec3 &v) const {
                return Vec3(x + v.x, y + v.y, z + v.z);
        }

        __host__ __device__ __forceinline__ Vec3 operator-(const Vec3 &v) const {
                return Vec3(x - v.x, y - v.y, z - v.z);
        }

        __host__ __device__ __forceinline__ Vec3 operator*(const float t) const {
                return Vec3(x * t, y * t, z * t);
        }

        __host__ __device__ __forceinline__ Vec3 operator*(const Vec3 v) const {
                return Vec3(x * v.x, y * v.y, z * v.z);
        }

        __host__ __device__ __forceinline__ Vec3 operator/(const float t) const {
                float inv_t = 1.0f / t;
                return Vec3(x * inv_t, y * inv_t, z * inv_t);
        }

        // length
        __host__ __device__ __forceinline__ float length_squared() const {
                return x * x + y * y + z * z;
        }

        __host__ __device__ __forceinline__ float length() const {
                return sqrtf(length_squared());
        }
};

// non-member functions
__host__ __device__ __forceinline__ float dot(const Vec3 &a, const Vec3 &b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ __forceinline__ Vec3 normalize(const Vec3 &v) {
        float len = v.length();
        return v / len;
}

__host__ __device__ __forceinline__ Vec3 cross(const Vec3 &a, const Vec3 &b) {
        return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

#endif  // VEC3_CUH
