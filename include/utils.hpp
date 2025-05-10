#pragma once

// ============================================================================
//  Macros
// ============================================================================

// Helper macro for inlining __host__ __device__ functions.
#ifdef __CUDACC__
#define CUDA_INLINE __host__ __device__ __forceinline__
#else
#define CUDA_INLINE inline
#endif

#include <curand_kernel.h>

// CUDA function call error detection.
#define CHECK_CUDA_ERROR(call)                                                                                         \
        {                                                                                                              \
                cudaError_t err = call;                                                                                \
                if (err != cudaSuccess) {                                                                              \
                        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__,                       \
                                cudaGetErrorString(err));                                                              \
                        cudaDeviceReset();                                                                             \
                        exit(EXIT_FAILURE);                                                                            \
                }                                                                                                      \
        }

// ============================================================================
//  Constants
// ============================================================================

const float FMAX = std::numeric_limits<float>::max();
const float PI   = 3.141592653589f;
const float TAU  = PI * 2;

// ============================================================================
//  Angle conversion
// ============================================================================
CUDA_INLINE float to_degrees(float radians) {
        return radians * 180.0f / PI;
}

CUDA_INLINE float to_radians(float degrees) {
        return degrees * PI / 180.0f;
}

// ============================================================================
//  Random number generation
// ============================================================================
#ifdef __CUDACC__

// Random float in ]0,1].
__device__ __forceinline__ float randf(curandState *rand_state) {
        return curand_uniform(rand_state);
}

// Random float in ]min, max].
__device__ __forceinline__ float randf(float min, float max, curandState *state) {
        return min + (max - min) * randf(state);
}

#else

// Random float in [0, 1].
inline float randf() {
        return (float)rand() / RAND_MAX;
}

// Random float in [min, max].
inline float randf(float min, float max) {
        return min + (max - min) * randf();
}

#endif

// ============================================================================
//  Intervals
// ============================================================================

// Clamp value x between min and max.
CUDA_INLINE float clamp(float x, float min, float max) {
        return x < min ? min : (x > max ? max : x);
}
