#include "texture.cuh"

#include "utils.cuh"

__device__ Vec3 Texture::value(float u, float v, const Vec3 p) const {
        switch (type) {
                case SOLID  : return color;
                case CHECKER: {
                        float sines = sinf(scale * p.x) * sinf(scale * p.y) * sinf(scale * p.z);
                        return sines < 0 ? color : color2;
                }
                case IMAGE: {
                        float4 c = tex2D<float4>(tex_obj, u, v);
                        return Vec3(c.x, c.y, c.z);
                }
                default: return Vec3(0, 0, 0);
        }
}

__host__ Texture texture_solid(const Vec3 &c) {
        Texture t = Texture();
        t.type    = SOLID;
        t.color   = c;
        return t;
}

__host__ Texture texture_checker(const Vec3 &c, const Vec3 &c2, float s) {
        Texture t;
        t.type   = CHECKER;
        t.color  = c;
        t.color2 = c2;
        t.scale  = s;
        return t;
}

__host__ Texture texture_image(const char path[]) {
        Texture t;
        t.type = IMAGE;

        int w, h, channels;
        unsigned char *data = stbi_load(path, &w, &h, &channels, 4);
        if (!data) {
                fprintf(stderr, "Failed to load image: %s", path);
                return t;
        }

        // rgba channel format
        const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        // copy to device
        cudaArray *d_array;
        CHECK_CUDA_ERROR(cudaMallocArray(&d_array, &channel_desc, w, h));
        const int array_size = w * h * 4 * sizeof(unsigned char);
        CHECK_CUDA_ERROR(cudaMemcpyToArray(d_array, 0, 0, data, array_size, cudaMemcpyHostToDevice));

        // resource descriptor
        cudaResourceDesc res_desc{};
        res_desc.resType         = cudaResourceTypeArray;
        res_desc.res.array.array = d_array;

        // texture descriptor
        cudaTextureDesc tex_desc{};
        tex_desc.addressMode[0]   = cudaAddressModeClamp;         // wrap horizontally
        tex_desc.addressMode[1]   = cudaAddressModeClamp;         // clamp vertically
        tex_desc.filterMode       = cudaFilterModeLinear;         // linear interpolation
        tex_desc.readMode         = cudaReadModeNormalizedFloat;  // return normalized floats [0,1]
        tex_desc.normalizedCoords = 1;                            // use [0,1] coordinates

        // texture object
        CHECK_CUDA_ERROR(cudaCreateTextureObject(&t.tex_obj, &res_desc, &tex_desc, nullptr));

        // free host memory
        stbi_image_free(data);

        return t;
}
