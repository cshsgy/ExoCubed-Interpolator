Interpolator of canoe ExoCubed data
Install:
Under the folder `interpolator`, run:
```
pip install .
```

Usage:
```
import exocubed_interpolator as interp

# input is a 4D tensor of shape (6, n_lyr, N*2, N*3)
# output is a 4D tensor of shape (6, n_lyr, n_lat, n_lon)
output = interp.cubed_to_latlon_interpolate(input, n_lat, n_lon)
```
Notice here n_lyr is a degenerate dimension, it can include layers, different variables, etc.
