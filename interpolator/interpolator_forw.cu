#include "interpolator.cuh"
#define MAX_N_LON 1024

__global__ void interpolate_kernel(
    const PRECISION* input,      // Input tensor (6*n_lyr*N*N)
    PRECISION* output,           // Output tensor (n_lyr*n_lat*n_lon)
    const int n_lyr,         // Number of layers
    const int N,             // Input grid size per face (N x N)
    const int n_lat,         // Number of latitude points
    const int n_lon         // Number of longitude points
){
    int64_t block_idx = blockIdx.x;
    int64_t thread_idx = threadIdx.x;
    if (block_idx >= n_lyr * n_lat or thread_idx >= n_lon) return;

    int64_t lyr = block_idx / n_lat;
    int64_t lat = block_idx % n_lat;
    int64_t lon = thread_idx;
    PRECISION lat_val = lat * M_PI / (n_lat - 1) - M_PI / 2;
    PRECISION lon_val = lon * 2 * M_PI / (n_lon - 1) - M_PI;

    // TODO: calculate the face
    // Second step: do the rotation
    // Then the input index. May be tricky at the borders and corners.
    // As a first approximation, we can just use (two) nearest neighbors for border points.

    int64_t out_idx = lyr * n_lat * n_lon + lat * n_lon + lon;
    output[out_idx] = input[out_idx];
}

torch::Tensor cubed_to_latlon_interpolate(
    const torch::Tensor& input,
    const int n_lat,
    const int n_lon
){
    TORCH_CHECK(input.dim() == 4, "Input tensor must have 4 dimensions");
    TORCH_CHECK(input.size(0) == 6, "Input tensor must have 6 layers");

    int64_t n_lyr = input.size(1);
    int64_t N = input.size(2);

    TORCH_CHECK(N == input.size(3), "Input tensor must have N x N grid size per face");
    TORCH_CHECK(n_lon <= MAX_N_LON, "Support for more than 1024 longitude points is not implemented");

    // Create output tensor
    torch::Tensor output = torch::zeros({n_lyr, n_lat, n_lon}, input.options());

    // Call the kernel
    int64_t block_dim = 1;
    while (block_dim < n_lon) block_dim *= 2;
    int64_t n_blocks = n_lyr * n_lat;
    interpolate_kernel<<<n_blocks, block_dim>>>(
        TensorWrapper(input).ptr(), 
        TensorWrapper(output).ptr(), 
        n_lyr, N, n_lat, n_lon);

    return output;
}