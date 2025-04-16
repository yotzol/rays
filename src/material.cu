#include "material.cuh"

Material lambertian_new(Texture texture) {
        Material m           = Material();
        m.type               = LAMBERTIAN;
        m.lambertian.texture = texture;
        return m;
}

Material metal_new(Vec3 albedo, float fuzz) {
        Material m     = Material();
        m.type         = METAL;
        m.metal.albedo = albedo;
        m.metal.fuzz   = fuzz;
        return m;
}

Material dielectric_new(float refraction_index) {
        Material m           = Material();
        m.type               = DIELECTRIC;
        m.dielectric.ref_idx = refraction_index;
        return m;
}

Material diffuse_light_new(Texture texture) {
        Material m              = Material();
        m.type                  = DIFFUSE_LIGHT;
        m.diffuse_light.texture = texture;
        return m;
}
