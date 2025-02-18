import math
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

def latlon_to_gnomonic_cubedsphere(lat_deg, lon_deg):
    """
    Convert (latitude, longitude) in degrees to
    - face index on the cubed sphere (0 through 5)
    - gnomonic equiangular coordinates (alpha, beta) in radians

    Face numbering convention used here:
      0 -> +X  1 -> -X
      2 -> +Y  3 -> -Y
      4 -> +Z  5 -> -Z
    """
    # 1. Convert degrees to radians
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    
    # 2. Convert spherical (lat, lon) to Cartesian (x, y, z) on the unit sphere
    x = math.cos(lat) * math.cos(lon)
    y = math.cos(lat) * math.sin(lon)
    z = math.sin(lat)
    
    # 3. Determine which face of the cube
    absx, absy, absz = abs(x), abs(y), abs(z)
    
    if absx >= absy and absx >= absz:
        # X-face
        if x > 0:
            face = 0  # +X
            # Intersection on plane x=1 => (1, y/x, z/x)
            alpha = -math.atan2(z, x)  # z/x
            beta  = -math.atan2(y, x)  # y/x
        else:
            face = 1  # -X
            # Intersection on plane x=-1 => (-1, y/(-x), z/(-x)) => effectively (-(y/x), -(z/x))
            alpha = math.atan2(-z, -x)  # -z/-x = z/x, but consistent sign with face
            beta  = -math.atan2(-y, -x)  # -y/-x = y/x
    elif absy >= absx and absy >= absz:
        # Y-face
        if y > 0:
            face = 2  # +Y
            # Intersection on plane y=1 => (x/y, 1, z/y)
            alpha = -math.atan2(z, y)  # z/y
            beta  = math.atan2(x, y)  # x/y
        else:
            face = 3  # -Y
            # Intersection on plane y=-1 => (x/(-y), -1, z/(-y))
            alpha = math.atan2(-z, -y)
            beta  = math.atan2(-x, -y)
    else:
        # Z-face
        if z > 0:
            face = 4  # +Z
            # Intersection on plane z=1 => (x/z, y/z, 1)
            alpha  = math.atan2(x, z)  # x/z
            beta = -math.atan2(y, z)  # y/z
        else:
            face = 5  # -Z
            # Intersection on plane z=-1 => (x/(-z), y/(-z), -1)
            alpha  = math.atan2(-x, -z)
            beta = math.atan2(-y, -z)
    
    return face, alpha, beta

def alphabeta_to_indices(alpha, beta, N):
    # N is the number of points per direction on a face
    # alpha and beta are the gnomonic coordinates
    # return the indices of the 4 nearest neighbors
    assert alpha >= -math.pi / 2 and alpha <= math.pi / 2
    assert beta >= -math.pi / 2 and beta <= math.pi / 2
    assert N % 2 == 0

    dalpha = math.pi / 2 / N
    id_alpha = alpha / dalpha + N / 2 - 0.5
    id_beta = beta / dalpha + N / 2 - 0.5
    id_alphas = [int(id_alpha - 0.5), int(id_alpha + 0.5)]
    id_betas = [int(id_beta - 0.5), int(id_beta + 0.5)]
    if id_alpha < 0.5:
        id_alphas[0] = 0
        id_alphas[1] = 1
    if id_alpha >= N - 0.5:
        id_alphas[1] = N - 1
        id_alphas[0] = N - 2
    if id_beta < 0.5:
        id_betas[0] = 0
        id_betas[1] = 1
    if id_beta >= N - 0.5:
        id_betas[1] = N - 1
        id_betas[0] = N - 2
    return id_alphas, id_betas, id_alpha, id_beta

def interpolate_value(value, id_alphas, id_betas, id_alpha, id_beta):
    # Bilinear interpolation weights
    w00 = (id_alphas[1] - id_alpha) * (id_betas[1] - id_beta)
    w01 = (id_alphas[1] - id_alpha) * (id_beta - id_betas[0]) 
    w10 = (id_alpha - id_alphas[0]) * (id_betas[1] - id_beta)
    w11 = (id_alpha - id_alphas[0]) * (id_beta - id_betas[0])
    return value[id_alphas[0], id_betas[0]] * w00 + \
           value[id_alphas[0], id_betas[1]] * w01 + \
           value[id_alphas[1], id_betas[0]] * w10 + \
           value[id_alphas[1], id_betas[1]] * w11

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
    
if __name__ == "__main__":
    # For example: lat = 45 N, lon = -90 E (i.e., lat=45, lon=-90)
    face_id, alpha, beta = latlon_to_gnomonic_cubedsphere(45.0, -90.0)
    id_alphas, id_betas, _, _ = alphabeta_to_indices(alpha, beta, 10)
    print(id_alphas, id_betas)

    # Load the dataset
    ds = xr.open_dataset('W92_single.nc')

    lat = np.squeeze(ds['lat'].values[:,0,:,:])
    lon = np.squeeze(ds['lon'].values[:,0,:,:])

    lat = torch.from_numpy(lat)
    lon = torch.from_numpy(lon)

    lat = exocubed_reshaping(lat.unsqueeze(0))
    lon = exocubed_reshaping(lon.unsqueeze(0))

    print(lat.shape)
    print(lon.shape)
    total_incorrect_count = 0
    N = lat.shape[2]
    for i in range(6):
        print(f"Face {i}")
        incorrect_count = 0
        incorrect_indices = []
        for j in range(N):
            for k in range(N):
                face_id, alpha, beta = latlon_to_gnomonic_cubedsphere(lat[0, i, j, k], lon[0, i, j, k])
                if face_id != i:
                    incorrect_count += 1
                    incorrect_indices.append(face_id)
        total_incorrect_count += incorrect_count
        print(f"Incorrect count: {incorrect_count}")
        print(f"Incorrect indices: {np.unique(incorrect_indices)}")
    assert total_incorrect_count == 0
    print(f"Face id check passed!")

    # Check the indices
    max_error = 0
    for i in range(6):
        for j in range(48):
            for k in range(48):
                face_id, alpha, beta = latlon_to_gnomonic_cubedsphere(lat[0, i, j, k], lon[0, i, j, k])
                id_alphas, id_betas, id_alpha, id_beta = alphabeta_to_indices(alpha, beta, 48)
                error = abs(j - id_alpha) + abs(k - id_beta)
                max_error = max(max_error, error)
                if j not in id_alphas or k not in id_betas:
                    print(alpha, beta)
                    raise ValueError(f"Mismatch at ({j}, {k}) on face {i}, expected {j}, {k}, got {id_alphas}, {id_betas}")
    print(f"Indices check passed!")
    print(f"Max error: {max_error}")
    # Example interpolation of lat and lon, and plot
    N_pts = 100
    lat_grid = np.linspace(-90, 90, N_pts)
    lon_grid = np.linspace(-180, 180, N_pts * 2)
    plt.figure(figsize=(10, 10))
    print("Interpolating...")
    all_lat = []
    all_lon = []
    all_lat_interp = []
    all_lon_interp = []
    for i in tqdm(range(N_pts)):
        for j in range(N_pts * 2):
            face_id, alpha, beta = latlon_to_gnomonic_cubedsphere(lat_grid[i], lon_grid[j])
            id_alphas, id_betas, id_alpha, id_beta = alphabeta_to_indices(alpha, beta, 48)
            interpolated_lat = interpolate_value(lat[0, face_id], id_alphas, id_betas, id_alpha, id_beta)
            all_lat.append(lat_grid[i])
            all_lat_interp.append(interpolated_lat)
            if abs(all_lat[-1] - all_lat_interp[-1]) > 3.0:
                print(id_alphas, id_betas, id_alpha, id_beta)
                print(lat[0, face_id, id_alphas[0], id_betas[0]], lat[0, face_id, id_alphas[0], id_betas[1]], lat[0, face_id, id_alphas[1], id_betas[0]], lat[0, face_id, id_alphas[1], id_betas[1]])
                raise ValueError(f"Mismatch at ({i}, {j}), expected {all_lat[-1]}, got {all_lat_interp[-1]}")
    plt.scatter(all_lat, all_lat_interp, c='blue')
    plt.xlabel('Original Latitude')
    plt.ylabel('Interpolated Latitude')
    plt.title('Latitude Interpolation')
    # Longitude is problematic because it is not a continuous function
    plt.savefig('interpolation.png')
            