#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "interp_forward.cuh"

namespace py = pybind11;

torch::Tensor exo_to_latlon(torch::Tensor input, int n_lat, int n_lon) {
    return cubed_to_latlon(input, n_lat, n_lon);
}

PYBIND11_MODULE(interpolator, m) {
    m.def("exo_to_latlon", &exo_to_latlon, "ExoCubed to latlon interpolation",
        py::arg("input"),py::arg("n_lat"),py::arg("n_lon"));
}