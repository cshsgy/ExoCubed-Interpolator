import math

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
            alpha = math.atan2(z, x)  # z/x
            beta  = math.atan2(y, x)  # y/x
        else:
            face = 1  # -X
            # Intersection on plane x=-1 => (-1, y/(-x), z/(-x)) => effectively (-(y/x), -(z/x))
            alpha = math.atan2(-z, -x)  # -z/-x = z/x, but consistent sign with face
            beta  = math.atan2(-y, -x)  # -y/-x = y/x
    elif absy >= absx and absy >= absz:
        # Y-face
        if y > 0:
            face = 2  # +Y
            # Intersection on plane y=1 => (x/y, 1, z/y)
            alpha = math.atan2(z, y)  # z/y
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
            alpha = math.atan2(y, z)  # y/z
            beta  = math.atan2(x, z)  # x/z
        else:
            face = 5  # -Z
            # Intersection on plane z=-1 => (x/(-z), y/(-z), -1)
            alpha = math.atan2(-y, -z)
            beta  = math.atan2(-x, -z)
    
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
    if id_alpha < 0:
        id_alphas[0] = 0
        id_alphas[1] = 1
    if id_alpha >= N:
        id_alphas[1] = N - 1
        id_alphas[0] = N - 2
    if id_beta < 0:
        id_betas[0] = 0
        id_betas[1] = 1
    if id_beta >= N:
        id_betas[1] = N - 1
        id_betas[0] = N - 2
    return id_alphas, id_betas




if __name__ == "__main__":
    # For example: lat = 45 N, lon = -90 E (i.e., lat=45, lon=-90)
    face_id, alpha, beta = latlon_to_gnomonic_cubedsphere(45.0, -90.0)
    id_alphas, id_betas = alphabeta_to_indices(alpha, beta, 10)
    print(id_alphas, id_betas)
