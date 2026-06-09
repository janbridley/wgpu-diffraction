#ifndef OCL_DEVICE_H
#define OCL_DEVICE_H

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct OCLDevice {
  cl_platform_id platform;
  cl_device_id device;
};

inline OCLDevice select_opencl_device() {
  cl_uint num_platforms;
  if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS ||
      num_platforms == 0)
    throw std::runtime_error("OpenCL: No platforms found.");

  std::vector<cl_platform_id> platforms(num_platforms);
  clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

  struct Candidate {
    cl_platform_id platform;
    cl_device_id device;
    int score;
    cl_ulong mem;
  };
  std::vector<Candidate> candidates;

  for (auto plat : platforms) {
    cl_uint nd;
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) !=
            CL_SUCCESS ||
        nd == 0)
      continue;
    std::vector<cl_device_id> devs(nd);
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);

    for (auto dev : devs) {
      cl_device_type type = CL_DEVICE_TYPE_DEFAULT;
      clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);

      cl_bool unified = CL_FALSE;
      clGetDeviceInfo(dev, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified),
                      &unified, nullptr);

      cl_ulong global_mem = 0;
      clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem),
                      &global_mem, nullptr);

      int score;
      if (type & CL_DEVICE_TYPE_GPU)
        score = unified ? 3 : 4;
      else if (type & CL_DEVICE_TYPE_CPU)
        score = 2;
      else
        score = 1;

      candidates.push_back({plat, dev, score, global_mem});
    }
  }

  if (candidates.empty())
    throw std::runtime_error("OpenCL: No compute devices found.");

  auto best = std::max_element(
      candidates.begin(), candidates.end(),
      [](const Candidate &a, const Candidate &b) {
        return std::tie(a.score, a.mem) < std::tie(b.score, b.mem);
      });

  return {best->platform, best->device};
}

#endif // OCL_DEVICE_H
