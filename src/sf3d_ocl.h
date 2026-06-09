#ifndef SF3D_OCL_H
#define SF3D_OCL_H

#include "ocl_device.h"

#include <stdexcept>
#include <string>
#include <vector>

class SF3DOpenCL {
public:
  SF3DOpenCL() {
    cl_int err;
    auto [platform, device] = select_opencl_device();
    m_device = device;

    m_context = clCreateContext(nullptr, 1, &m_device, nullptr, nullptr, &err);
    check(err, "Failed to create context.");
    m_queue = clCreateCommandQueue(m_context, m_device, 0, &err);
    check(err, "Failed to create command queue.");

    const char *src = kernel_source;
    m_program = clCreateProgramWithSource(m_context, 1, &src, nullptr, &err);
    check(err, "Failed to create program.");
    if (clBuildProgram(m_program, 1, &m_device, nullptr, nullptr, nullptr) !=
        CL_SUCCESS) {
      size_t log_size;
      clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, 0,
                            nullptr, &log_size);
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, log_size,
                            log.data(), nullptr);
      throw std::runtime_error("OpenCL Build Error:\n" +
                               std::string(log.data()));
    }
    m_kernel = clCreateKernel(m_program, "compute_sf3d", &err);
    check(err, "Failed to create kernel.");
  }

  ~SF3DOpenCL() {
    if (m_kernel)
      clReleaseKernel(m_kernel);
    if (m_program)
      clReleaseProgram(m_program);
    if (m_queue)
      clReleaseCommandQueue(m_queue);
    if (m_context)
      clReleaseContext(m_context);
  }

  void compute(const float *points, const float *k_vecs, unsigned int N,
               unsigned int Nk, float *out) {
    Buffer bufPoints(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * 3 * N, points);
    Buffer bufK(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * 3 * Nk, k_vecs);
    Buffer bufOut(m_context, CL_MEM_WRITE_ONLY, sizeof(float) * Nk, nullptr);

    clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &bufPoints.mem);
    clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &bufK.mem);
    clSetKernelArg(m_kernel, 2, sizeof(cl_mem), &bufOut.mem);
    clSetKernelArg(m_kernel, 3, sizeof(unsigned int), &N);
    clSetKernelArg(m_kernel, 4, sizeof(unsigned int), &Nk);

    size_t global_size = Nk;
    check(clEnqueueNDRangeKernel(m_queue, m_kernel, 1, nullptr, &global_size,
                                 nullptr, 0, nullptr, nullptr),
          "Kernel execution failed.");
    clEnqueueReadBuffer(m_queue, bufOut.mem, CL_TRUE, 0, sizeof(float) * Nk,
                        out, 0, nullptr, nullptr);
  }

private:
  struct Buffer {
    cl_mem mem;
    Buffer(cl_context ctx, cl_mem_flags flags, size_t size,
           const void *host_ptr) {
      cl_int err;
      mem =
          clCreateBuffer(ctx, flags, size, const_cast<void *>(host_ptr), &err);
      if (err != CL_SUCCESS)
        throw std::runtime_error("OpenCL: Failed to allocate GPU memory.");
    }
    ~Buffer() { clReleaseMemObject(mem); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
  };

  static void check(cl_int err, const char *msg) {
    if (err != CL_SUCCESS)
      throw std::runtime_error(std::string("OpenCL: ") + msg);
  }

  cl_device_id m_device{nullptr};
  cl_context m_context{nullptr};
  cl_command_queue m_queue{nullptr};
  cl_program m_program{nullptr};
  cl_kernel m_kernel{nullptr};

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
