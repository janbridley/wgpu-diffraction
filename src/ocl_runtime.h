#ifndef OCL_RUNTIME_H
#define OCL_RUNTIME_H

#include "ocl_device.h"

#include <stdexcept>
#include <string>
#include <vector>

class OCLRuntime {
public:
  OCLRuntime(const char *kernel_name, const char *source) {
    cl_int err;
    auto [platform, device] = select_opencl_device();
    m_device = device;

    m_context =
        clCreateContext(nullptr, 1, &m_device, nullptr, nullptr, &err);
    check(err, "Failed to create context.");
    m_queue = clCreateCommandQueue(m_context, m_device, 0, &err);
    check(err, "Failed to create command queue.");

    m_program =
        clCreateProgramWithSource(m_context, 1, &source, nullptr, &err);
    check(err, "Failed to create program.");
    if (clBuildProgram(m_program, 1, &m_device, nullptr, nullptr, nullptr) !=
        CL_SUCCESS) {
      size_t log_size;
      clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, 0,
                            nullptr, &log_size);
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG,
                            log_size, log.data(), nullptr);
      throw std::runtime_error("OpenCL Build Error:\n" +
                               std::string(log.data()));
    }
    m_kernel = clCreateKernel(m_program, kernel_name, &err);
    check(err, "Failed to create kernel.");
  }

  ~OCLRuntime() {
    if (m_kernel)
      clReleaseKernel(m_kernel);
    if (m_program)
      clReleaseProgram(m_program);
    if (m_queue)
      clReleaseCommandQueue(m_queue);
    if (m_context)
      clReleaseContext(m_context);
  }

  OCLRuntime(const OCLRuntime &) = delete;
  OCLRuntime &operator=(const OCLRuntime &) = delete;

  struct Buffer {
    cl_mem mem{nullptr};
    Buffer() = default;
    explicit Buffer(cl_mem m) : mem(m) {}
    ~Buffer() {
      if (mem)
        clReleaseMemObject(mem);
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
  };

  Buffer create_buffer(cl_mem_flags flags, size_t size, const void *data) {
    cl_int err;
    cl_mem m = clCreateBuffer(m_context, flags, size,
                              const_cast<void *>(data), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("OpenCL: Failed to allocate device memory.");
    return Buffer(m);
  }

  template <typename T> void set_arg(unsigned int index, const T &value) {
    check(clSetKernelArg(m_kernel, index, sizeof(T), &value),
          "Failed to set kernel argument.");
  }

  void run(size_t global_size) {
    check(clEnqueueNDRangeKernel(m_queue, m_kernel, 1, nullptr, &global_size,
                                 nullptr, 0, nullptr, nullptr),
          "Kernel execution failed.");
  }

  void read(const Buffer &buf, size_t size, void *out) {
    clEnqueueReadBuffer(m_queue, buf.mem, CL_TRUE, 0, size, out, 0, nullptr,
                        nullptr);
  }

private:
  static void check(cl_int err, const char *msg) {
    if (err != CL_SUCCESS)
      throw std::runtime_error(std::string("OpenCL: ") + msg);
  }

  cl_device_id m_device{nullptr};
  cl_context m_context{nullptr};
  cl_command_queue m_queue{nullptr};
  cl_program m_program{nullptr};
  cl_kernel m_kernel{nullptr};
};

#endif // OCL_RUNTIME_H
