#pragma once
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "tensor_wrapper.hpp"

#ifndef PRECISION
#define PRECISION double
#endif


// Interpolation kernel declaration
__global__ void interpolate_kernel(
    const PRECISION* input,      // Input tensor (6*n_lyr*N*N)
    PRECISION* output,           // Output tensor (n_lyr*n_lat*n_lon)
    const int n_lyr,         // Number of layers
    const int N,             // Input grid size per face (N x N)
    const int n_lat,         // Number of latitude points
    const int n_lon         // Number of longitude points
);

// Host function declaration that will call the kernel
torch::Tensor cubed_to_latlon(
    const torch::Tensor& input,
    const int n_lat,
    const int n_lon
);
