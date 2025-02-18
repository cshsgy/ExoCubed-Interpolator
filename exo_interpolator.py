import torch
import math
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt

def exocubed_reshaping(data: torch.Tensor):
    # data is a 3D tensor of shape (n_lyr, N*2, N*3)
    # we want to reshape it to a 4D tensor of shape (n_lyr, 6, N, N)
    # Excubed layout consistent with the above notation:
    #| 4 | 3 | 5 |
    #-------------
    #| 0 | 2 | 1 |
    N = data.shape[1] // 2
    output = torch.empty((data.shape[0], 6, N, N))
    output[:, 0, :, :] = data[:, N:2*N, :N]
    output[:, 1, :, :] = data[:, N:2*N, 2*N:3*N]
    output[:, 2, :, :] = data[:, :N, N:2*N]
    output[:, 3, :, :] = data[:, N:2*N, N:2*N]
    output[:, 4, :, :] = data[:, :N, :N]
    output[:, 5, :, :] = data[:, :N, 2*N:3*N]
    return output


def latlon_to_gnomonic_cubedsphere(lat_deg, lon_deg):
    """
    Convert (latitude, longitude) in degrees to
    - face index on the cubed sphere (0 through 5)
    - gnomonic equiangular coordinates (alpha, beta) in radians

    Face numbering convention used here:
      0 -> +X  1 -> -X
      2 -> +Y  3 -> -Y
      4 -> +Z  5 -> -Z

    Note: both inputs are tensors of shape (n_lyr, n_lat, n_lon)
    """
    # 1. Convert degrees to radians
    lat = torch.deg2rad(lat_deg)
    lon = torch.deg2rad(lon_deg)
    
    # 2. Convert spherical (lat, lon) to Cartesian (x, y, z) on the unit sphere
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    
    # 3. Determine which face of the cube
    absx, absy, absz = torch.abs(x), torch.abs(y), torch.abs(z)

    # 4. Determine the face and the gnomonic coordinates
    face = torch.empty_like(x)
    alpha = torch.empty_like(x)
    beta = torch.empty_like(x)
    # +X face
    tmp_idx = torch.where((absx >= absy) & (absx >= absz) & (x > 0))
    face[tmp_idx] = 0
    alpha[tmp_idx] = -torch.atan2(z[tmp_idx], x[tmp_idx])
    beta[tmp_idx] = -torch.atan2(y[tmp_idx], x[tmp_idx])
    # -X face
    tmp_idx =  torch.where((absx >= absy) & (absx >= absz) & (x < 0))
    face[tmp_idx] = 1
    alpha[tmp_idx] = torch.atan2(-z[tmp_idx], -x[tmp_idx])
    beta[tmp_idx] = -torch.atan2(-y[tmp_idx], -x[tmp_idx]) # One added minus sign, to be checked
    # +Y face
    tmp_idx =  torch.where((absy >= absx) & (absy >= absz) & (y > 0))
    face[tmp_idx] = 2
    alpha[tmp_idx] = -torch.atan2(z[tmp_idx], y[tmp_idx])
    beta[tmp_idx] = torch.atan2(x[tmp_idx], y[tmp_idx])
    # -Y face
    tmp_idx =  torch.where((absy >= absx) & (absy >= absz) & (y < 0))
    face[tmp_idx] = 3
    alpha[tmp_idx] = torch.atan2(-z[tmp_idx], -y[tmp_idx])
    beta[tmp_idx] = torch.atan2(-x[tmp_idx], -y[tmp_idx])
    # +Z face
    tmp_idx =  torch.where((absz >= absx) & (absz >= absy) & (z > 0))
    face[tmp_idx] = 4
    alpha[tmp_idx] = torch.atan2(x[tmp_idx], z[tmp_idx])
    beta[tmp_idx] = -torch.atan2(y[tmp_idx], z[tmp_idx])
    # -Z face
    tmp_idx =  torch.where((absz >= absx) & (absz >= absy) & (z < 0))
    face[tmp_idx] = 5
    alpha[tmp_idx] = torch.atan2(-x[tmp_idx], -z[tmp_idx])
    beta[tmp_idx] = torch.atan2(-y[tmp_idx], -z[tmp_idx])
    
    return face, alpha, beta

def alphabeta_to_indices(alpha, beta, N):
    # N is the number of points per direction on a face
    # alpha and beta are tensors of shape (n_lyr, n_lat, n_lon)
    assert torch.all(alpha >= -math.pi / 2) and torch.all(alpha <= math.pi / 2)
    assert torch.all(beta >= -math.pi / 2) and torch.all(beta <= math.pi / 2)
    assert N % 2 == 0

    dalpha = math.pi / 2 / N
    id_alpha = alpha / dalpha + N / 2 - 0.5
    id_beta = beta / dalpha + N / 2 - 0.5
    id_alpha_0 = torch.floor(id_alpha - 0.5).to(torch.int64)
    id_alpha_1 = torch.floor(id_alpha + 0.5).to(torch.int64)
    id_beta_0 = torch.floor(id_beta - 0.5).to(torch.int64)
    id_beta_1 = torch.floor(id_beta + 0.5).to(torch.int64)
    tmp_idx = torch.where(id_alpha < 0.5)
    id_alpha_0[tmp_idx] = 0
    id_alpha_1[tmp_idx] = 1
    tmp_idx = torch.where(id_alpha >= N - 0.5)
    id_alpha_0[tmp_idx] = N - 1
    id_alpha_1[tmp_idx] = N - 2
    tmp_idx = torch.where(id_beta < 0.5)
    id_beta_0[tmp_idx] = 0
    id_beta_1[tmp_idx] = 1
    tmp_idx = torch.where(id_beta >= N - 0.5)
    id_beta_0[tmp_idx] = N - 1
    id_beta_1[tmp_idx] = N - 2
    return [id_alpha_0, id_alpha_1], [id_beta_0, id_beta_1], id_alpha, id_beta

def interpolate_value(value, face, id_alphas, id_betas, id_alpha, id_beta):
    # Input: value: (n_lyr, 6, N, N)
    # face: (n_lyr, nlat, nlon)
    # id_alphas: list of 2 tensors, each (n_lyr, nlat, nlon)
    # id_betas: list of 2 tensors, each (n_lyr, nlat, nlon)
    # id_alpha: (n_lyr, nlat, nlon)
    # id_beta: (n_lyr, nlat, nlon)
    # Output: value_interp: (n_lyr, nlat, nlon)
    
    # Flatten all coordinate tensors, ensure can be used as indices
    flat_face = face.flatten().to(torch.int64)
    flat_alpha0 = id_alphas[0].flatten().to(torch.int64)
    flat_alpha1 = id_alphas[1].flatten().to(torch.int64)
    flat_beta0 = id_betas[0].flatten().to(torch.int64)
    flat_beta1 = id_betas[1].flatten().to(torch.int64)

    # Calculate weights
    w00 = ((id_alphas[1] - id_alpha) * (id_betas[1] - id_beta)).flatten()
    w01 = ((id_alphas[1] - id_alpha) * (id_beta - id_betas[0])).flatten()
    w10 = ((id_alpha - id_alphas[0]) * (id_betas[1] - id_beta)).flatten()
    w11 = ((id_alpha - id_alphas[0]) * (id_beta - id_betas[0])).flatten()

    result = (value[:, flat_face, flat_alpha0, flat_beta0] * w00 + 
             value[:, flat_face, flat_alpha0, flat_beta1] * w01 + 
             value[:, flat_face, flat_alpha1, flat_beta0] * w10 + 
             value[:, flat_face, flat_alpha1, flat_beta1] * w11)
    
    return result.reshape(value.shape[0], face.shape[1], face.shape[2])

def exo_to_latlon(exo_data, n_lat, n_lon):
    # exo_data is a 4D tensor of shape (n_lyr, 6, N, N)
    # n_lat and n_lon are the number of points per direction on the lat-lon grid
    # Output: latlon_data: (n_lyr, n_lat, n_lon)
    N = exo_data.shape[2]
    lat, lon = np.linspace(-90, 90, n_lat), np.linspace(-180, 180, n_lon)
    lat, lon = np.meshgrid(lat, lon)
    lat = np.transpose(lat)
    lon = np.transpose(lon)
    lat = np.tile(lat[None, :, :], (exo_data.shape[0], 1, 1))
    lon = np.tile(lon[None, :, :], (exo_data.shape[0], 1, 1))
    lat = torch.from_numpy(lat)
    lon = torch.from_numpy(lon)
    face, alpha, beta = latlon_to_gnomonic_cubedsphere(lat, lon)
    id_alphas, id_betas, id_alpha, id_beta = alphabeta_to_indices(alpha, beta, N)
    value_interp = interpolate_value(exo_data, face, id_alphas, id_betas, id_alpha, id_beta)
    return value_interp

if __name__ == "__main__":
    import time
    exo_data = xr.open_dataset('W92_single.nc')
    data = np.squeeze(exo_data['U'].values[:,0,:,:])
    data = torch.from_numpy(data).unsqueeze(0)
    n_lat, n_lon = 100, 200
    lat = np.linspace(-90, 90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    start_time = time.time()
    for i in range(100):
        latlon_data = exo_to_latlon(exocubed_reshaping(data), n_lat, n_lon)
        torch.cuda.synchronize()
    end_time = time.time()
    print(f"Average time taken: {(end_time - start_time) / 100} seconds")
    plt.imshow(latlon_data[0,:,:])
    plt.savefig('test_U_latlon.png')
