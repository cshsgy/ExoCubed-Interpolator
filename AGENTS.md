## Cursor Cloud specific instructions

This is a scientific Python/CUDA project for interpolating ExoCubed (cubed-sphere) atmospheric simulation data to standard lat-lon grids. See `README.md` for basic usage.

### Key points

- **No GPU in Cloud VM**: The CUDA C++ extension (`interpolator/`) cannot be built without an NVIDIA GPU. The pure-Python interpolator (`exo_interpolator.py`) works on CPU, but `latlon_from_nc()` and the `__main__` block call `.cuda()` directly. To test on CPU, call `exocubed_reshaping()` and `exo_to_latlon()` individually with CPU tensors instead.
- **No lockfile or requirements.txt**: Dependencies must be inferred from imports. Core: `torch`, `xarray`, `netCDF4`, `numpy`, `matplotlib`, `scipy`, `tqdm`, `imageio`.
- **PyTorch install**: Use `pip install torch --index-url https://download.pytorch.org/whl/cpu` for CPU-only environments, then install remaining packages from PyPI separately.
- **Test data**: `W92_single.nc` (single timestep) and `W92-main.nc` (6 timesteps) are committed in the repo root. `hotjupiter-main.nc` (referenced by `test_exo_3d.py`) is not included.
- **Sanity check**: `python python_sanity_check.py` is the primary validation script — it tests face-ID mapping, index computation, and interpolation accuracy against `W92_single.nc`. It runs entirely on CPU.
- **Lint**: `python -m pyflakes *.py` for basic static analysis. No project-specific lint config exists.
- **No formal test framework**: Tests are standalone scripts (`python_sanity_check.py`, `test_exo_3d.py`, `interpolator/test_interpolator.py`), not pytest-based.
