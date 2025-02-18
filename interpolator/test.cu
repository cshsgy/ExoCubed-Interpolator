#include <torch/torch.h>
#include <iostream>
#include "interpolator.cuh"

int main() {
    const int d_1 = 48;
    const int d_2 = 48;
    const int n = 20;
    const int n_lat = 50;
    const int n_lon = 50;
    torch::Tensor input = torch::randn({6, n, d_1, d_2}, torch::kFloat64).cuda();
    torch::Tensor output = cubed_to_latlon_interpolate(input, n_lat, n_lon);
    std::cout << "Output shape: " << output.sizes() << std::endl;
    return 0;
}


