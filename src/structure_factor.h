#ifndef SF3D_OCL_H
#define SF3D_OCL_H

#include "ocl_runtime.h"

#include <cstddef>

class SF3DOpenCL {
public:
  void compute(const float *points, const float *k_vecs, unsigned int N,
               unsigned int Nk, float *out) {
    auto bufP = m_rt.create_buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * 3 * N, points);
    auto bufK = m_rt.create_buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * 3 * Nk, k_vecs);
    auto bufO = m_rt.create_buffer(CL_MEM_WRITE_ONLY, sizeof(float) * Nk,
                                   nullptr);

    m_rt.set_arg(0, bufP.mem);
    m_rt.set_arg(1, bufK.mem);
    m_rt.set_arg(2, bufO.mem);
    m_rt.set_arg(3, N);
    m_rt.set_arg(4, Nk);

    m_rt.run(Nk);
    m_rt.read(bufO, sizeof(float) * Nk, out);
  }

private:
  OCLRuntime m_rt{"compute_sf3d", kernel_source};

  static constexpr const char *kernel_source = R"(
    __kernel void compute_sf3d(
        __global const float* points,
        __global const float* k_vecs,
        __global float* out,
        const unsigned int N,
        const unsigned int Nk)
    {
        int k_idx = get_global_id(0);
        if (k_idx >= Nk) return;

        float3 k = vload3(k_idx, k_vecs);
        if (dot(k, k) < 1e-10f) {
            out[k_idx] = 0.0f;
            return;
        }

        float sum_cos = 0.0f, sum_sin = 0.0f;
        for (unsigned int i = 0; i < N; i++) {
            float cos_val;
            float sin_val = sincos(dot(k, vload3(i, points)), &cos_val);
            sum_cos += cos_val;
            sum_sin += sin_val;
        }
        out[k_idx] = (sum_cos * sum_cos + sum_sin * sum_sin) / (float)N;
    })";
};

#endif // SF3D_OCL_H
