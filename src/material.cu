#include "material.cuh"

#include <cstdio>

Material lambertian_new(Texture texture) {
        Material m = Material();
        m.texture  = texture;
        return m;
}

Material metal_new(Vec3 albedo, float fuzz) {
        Material m = Material();
        m.type     = METAL;
        m.albedo   = albedo;
        m.fuzz     = fuzz;
        return m;
}

Material dielectric_new(float refraction_index) {
        Material m = Material();
        m.type     = DIELECTRIC;
        m.ref_idx  = refraction_index;
        return m;
}
