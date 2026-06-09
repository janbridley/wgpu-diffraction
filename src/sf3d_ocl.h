#ifndef SF3D_OCL_H
#define SF3D_OCL_H

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief RAII-style OpenCL implementation for the 3D Static Structure Factor.
 */
class SF3DOpenCL {
public:
  SF3DOpenCL() {
    cl_int err;
    cl_platform_id platform;
    if (clGetPlatformIDs(1, &platform, nullptr) != CL_SUCCESS) {
      throw std::runtime_error("OpenCL: Failed to find any platforms.");
    }
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &m_device, nullptr) !=
        CL_SUCCESS) {
      throw std::runtime_error("OpenCL: Failed to find a GPU device.");
    }
    m_context = clCreateContext(nullptr, 1, &m_device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("OpenCL: Failed to create context.");
    m_queue = clCreateCommandQueue(m_context, m_device, 0, &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("OpenCL: Failed to create command queue.");
    const char *src = kernel_source;
    m_program = clCreateProgramWithSource(m_context, 1, &src, nullptr, &err);
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
    if (err != CL_SUCCESS)
      throw std::runtime_error("OpenCL: Failed to create kernel.");
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
    cl_int err;
    cl_mem bufPoints =
        clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * 3 * N, (void *)points, &err);
    cl_mem bufK =
        clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * 3 * Nk, (void *)k_vecs, &err);
    cl_mem bufOut = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY,
                                   sizeof(float) * Nk, nullptr, &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("OpenCL: Failed to allocate GPU memory.");
    clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &bufPoints);
    clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &bufK);
    clSetKernelArg(m_kernel, 2, sizeof(cl_mem), &bufOut);
    clSetKernelArg(m_kernel, 3, sizeof(unsigned int), &N);
    clSetKernelArg(m_kernel, 4, sizeof(unsigned int), &Nk);
    size_t global_size = Nk;
    err = clEnqueueNDRangeKernel(m_queue, m_kernel, 1, nullptr, &global_size,
                                 nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      clReleaseMemObject(bufPoints);
      clReleaseMemObject(bufK);
      clReleaseMemObject(bufOut);
      throw std::runtime_error("OpenCL: Kernel execution failed.");
    }
    clEnqueueReadBuffer(m_queue, bufOut, CL_TRUE, 0, sizeof(float) * Nk, out, 0,
                        nullptr, nullptr);
    clReleaseMemObject(bufPoints);
    clReleaseMemObject(bufK);
    clReleaseMemObject(bufOut);
  }

private:
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
            if (k_idx >= Nk) { return };
            float kx = k_vecs[3 * k_idx + 0];
            float ky = k_vecs[3 * k_idx + 1];
            float kz = k_vecs[3 * k_idx + 2];

            // Mask out the k=0 peak
            if (kx*kx + ky*ky + kz*kz < 1e-10f) {
                out[k_idx] = 0.0f;
                return;
            }

            float sum_cos = 0.0f;

            float sum_sin = 0.0f;
            for (unsigned int i = 0; i < N; i++) {
                float phase = kx * points[3*i+0] + ky * points[3*i+1] + kz * points[3*i+2];
                sum_cos += cos(phase);
                sum_sin += sin(phase);
            }
            out[k_idx] = (sum_cos * sum_cos + sum_sin * sum_sin) / (float)N;
        }
    )";
};

#endif // SF3D_OCL_H
