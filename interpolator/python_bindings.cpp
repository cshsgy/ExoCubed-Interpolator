#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "interpolator.cuh"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cubed_to_latlon_interpolate", &cubed_to_latlon_interpolate, "Cubed to latlon interpolation",
        py::arg("input"),
        py::arg("n_lat"),
        py::arg("n_lon")
    );
}
