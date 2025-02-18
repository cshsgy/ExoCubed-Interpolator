#include "interp_forward.cuh"
#include "tensor_wrapper.hpp"
#include <cmath>

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

    PRECISION x = cos(lat_val) * cos(lon_val);
    PRECISION y = cos(lat_val) * sin(lon_val);
    PRECISION z = sin(lat_val);

    PRECISION absx = abs(x);
    PRECISION absy = abs(y);
    PRECISION absz = abs(z);

    int64_t face = 0;
    PRECISION alpha = 0;
    PRECISION beta = 0;

    if (absx >= absy and absx >= absz) {
        // X-face
        if (x > 0) {
            face = 0;  // +X
            // Intersection on plane x=1 => (1, y/x, z/x)
            alpha = -atan2(z, x);  // z/x
            beta  = -atan2(y, x);  // y/x
        } else {
            face = 1;  // -X
            // Intersection on plane x=-1 => (-1, y/(-x), z/(-x)) => effectively (-(y/x), -(z/x))
            alpha = atan2(-z, -x);  // -z/-x = z/x, but consistent sign with face
            beta  = -atan2(-y, -x);  // -y/-x = y/x
        }
    } else if (absy >= absx and absy >= absz) {
        // Y-face
        if (y > 0) {
            face = 2;  // +Y
            // Intersection on plane y=1 => (x/y, 1, z/y)
            alpha = -atan2(z, y);  // z/y
            beta  = atan2(x, y);  // x/y
        } else {
            face = 3;  // -Y
            // Intersection on plane y=-1 => (x/(-y), -1, z/(-y))
            alpha = atan2(-z, -y);
            beta  = atan2(-x, -y);
        }
    } else {
        // Z-face
        if (z > 0) {
            face = 4;  // +Z
            // Intersection on plane z=1 => (x/z, y/z, 1)
            alpha  = atan2(x, z);  // x/z
            beta = -atan2(y, z);  // y/z
        } else {
            face = 5;  // -Z
            // Intersection on plane z=-1 => (x/(-z), y/(-z), -1)
            alpha  = atan2(-x, -z);
            beta = atan2(-y, -z);
        }
    }

    PRECISION dalpha = M_PI / 2 / N;
    PRECISION id_alpha = alpha / dalpha + N / 2 - 0.5;
    PRECISION id_beta = beta / dalpha + N / 2 - 0.5;
    int64_t id_alphas[2] = {int(id_alpha - 0.5), int(id_alpha + 0.5)};
    int64_t id_betas[2] = {int(id_beta - 0.5), int(id_beta + 0.5)};
    if (id_alpha < 0.5) {
        id_alphas[0] = 0;
        id_alphas[1] = 1;
    }
    if (id_alpha >= N - 0.5) {
        id_alphas[1] = N - 1;
        id_alphas[0] = N - 2;
    }
    if (id_beta < 0.5) {
        id_betas[0] = 0;
        id_betas[1] = 1;
    }
    if (id_beta >= N - 0.5) {
        id_betas[1] = N - 1;
        id_betas[0] = N - 2;
    }
    PRECISION w00 = (id_alphas[1] - id_alpha) * (id_betas[1] - id_beta);
    PRECISION w01 = (id_alphas[1] - id_alpha) * (id_beta - id_betas[0]); 
    PRECISION w10 = (id_alpha - id_alphas[0]) * (id_betas[1] - id_beta);
    PRECISION w11 = (id_alpha - id_alphas[0]) * (id_beta - id_betas[0]);

    int64_t out_idx = lyr * n_lat * n_lon + lat * n_lon + lon;
    output[out_idx] = input[face * n_lyr * N * N + lyr * N * N + id_alphas[0] * N + id_betas[0]] * w00 + \
                       input[face * n_lyr * N * N + lyr * N * N + id_alphas[0] * N + id_betas[1]] * w01 + \
                       input[face * n_lyr * N * N + lyr * N * N + id_alphas[1] * N + id_betas[0]] * w10 + \
                       input[face * n_lyr * N * N + lyr * N * N + id_alphas[1] * N + id_betas[1]] * w11;
}

torch::Tensor cubed_to_latlon(
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