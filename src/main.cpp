#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "structure_factor.h"

namespace nb = nanobind;

static SF3DOpenCL engine;

nb::ndarray<nb::numpy, float, nb::shape<-1>>
sf3d(nb::ndarray<float, nb::shape<-1, 3>> points,
     nb::ndarray<float, nb::shape<-1, 3>> k_vecs) {
             
    size_t N = points.shape(0);
    size_t Nk = k_vecs.shape(0);
    
    // Allocate memory for the output
    float* data = new float[Nk];
    
    // Create a capsule to manage the memory
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (float *) p;
    });

    engine.compute((const float*)points.data(), (const float*)k_vecs.data(), (unsigned int)N, (unsigned int)Nk, data);
    
    return nb::ndarray<nb::numpy, float, nb::shape<-1>>(
        data, 1, &Nk, owner
    );
}

NB_MODULE(sf3d_gpu, m) {
    m.def("sf3d", &sf3d, "Compute 3D Structure Factor using OpenCL");
}
