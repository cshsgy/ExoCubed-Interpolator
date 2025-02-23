import torch
from math import pi

def spherical_vorticity_torch(u, v, R, n_lat, n_lon, planetary_omega=None):
    """
    Compute the '2D local normal' vorticity on a sphere using PyTorch tensors.
    input shape of u and v is (n_batch, n_lat, n_lon)
    """

    # Shape checks
    n_batch, n_lat, n_lon = u.shape
    assert v.shape == (n_batch, n_lat, n_lon), "v must match shape of u"

    # Convert to radians
    lat_rad = torch.linspace(-pi/2, pi/2, n_lat)  # shape (n_lat,)
    lon_rad = torch.linspace(-pi, pi, n_lon)  # shape (n_lon,)
    lat_rad_2D = lat_rad.view(1, n_lat, 1)    # (1, n_lat, 1)
    cos_phi    = torch.cos(lat_rad_2D)        # (1, n_lat, 1)
    sin_phi    = torch.sin(lat_rad_2D)        # (1, n_lat, 1)

    dphi = lat_rad[1] - lat_rad[0]   # spacing in latitude
    dlam = lon_rad[1] - lon_rad[0]   # spacing in longitude

    vcos = v * cos_phi               # shape = (n_batch, n_lat, n_lon)
    dvcos_dlam = torch.gradient(vcos, spacing=dlam, dim=2)[0]  # partial wrt axis=2
    du_dphi = torch.gradient(u, spacing=dphi, dim=1)[0]        # partial wrt axis=1
    # zeta_rel = (1 / (R cos(phi))) * [ dvcos_dlam - du_dphi ]
    zeta_rel = (dvcos_dlam - du_dphi) / (R * cos_phi)

    if planetary_omega is not None:
        zeta_abs = zeta_rel + 2.0 * planetary_omega * sin_phi
        return zeta_abs
    else:
        return zeta_rel


if __name__ == "__main__":
    # Example grid sizes
    n_batch = 1
    n_lat   = 91   # from -90 to +90
    n_lon   = 181  # from -180 to +180

    # Earth-like constants (just for demonstration)
    R_earth = 6.371e6             # Earth's radius in meters
    Omega   = 7.2921159e-5        # Earth's angular velocity [rad/s]

    # shape = (n_batch, n_lat, n_lon)
    u_test = torch.zeros((n_batch, n_lat, n_lon), dtype=torch.float64)
    v_test = torch.zeros((n_batch, n_lat, n_lon), dtype=torch.float64)

    zeta_abs = spherical_vorticity_torch(
        u_test, 
        v_test, 
        R=R_earth, 
        n_lat=n_lat, 
        n_lon=n_lon,
        planetary_omega=Omega
    )  # shape = (n_batch, n_lat, n_lon)

    # The expected formula: 2 * Omega * sin(phi)
    lat_rad = torch.linspace(-pi/2, pi/2, n_lat).to(dtype=torch.float64)  # shape (n_lat,)
    sin_phi = torch.sin(lat_rad).view(1, n_lat, 1)  # broadcast shape (1,n_lat,1)
    zeta_expected = 2.0 * Omega * sin_phi         # shape (1,n_lat,1)
    zeta_expected_2d = zeta_expected.expand(n_batch, n_lat, n_lon)

    # Compare with torch.allclose:
    are_close = torch.allclose(zeta_abs, zeta_expected_2d, rtol=1e-6, atol=1e-10)
    print("Do we get 2 * Omega * sin(phi) everywhere?", are_close)

    # Print some diagnostics
    if not are_close:
        diff = zeta_abs - zeta_expected_2d
        print("Max abs difference = ", diff.abs().max().item())
    else:
        print("Success! The computed vorticity matches 2 * Omega * sin(phi).")

    # Example numeric check at equator (phi=0) and north pole (phi=+90°):
    # - equator => sin(0)=0 => vorticity should be 0
    # - north pole => sin(90°)=1 => vorticity should be ~ 2*Omega = 1.4584e-4, showing up 0.0001
    print("\nSome sample values (all longitudes) at the equator (j ~ n_lat/2):")
    equator_idx = n_lat // 2
    print(zeta_abs[0, equator_idx, :10])  # first 10 longs
    print("\nValue near north pole (top index):")
    print(zeta_abs[0, -1, :10])
