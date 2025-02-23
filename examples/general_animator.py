from exo_interpolator import latlon_from_nc
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import torch
from other_tools.vorticity import spherical_vorticity_torch

def animator_from_folder(folder, pattern, step, output_name, nlat=300, nlon=600):
    # Get and sort files
    files = glob.glob(os.path.join(folder, pattern))
    files.sort()
    
    # Create frames list to store images
    frames = []
    
    # Get the color limits
    data_u = latlon_from_nc(files[step-1], "U", nlat, nlon, is_2D=True)
    data_v = latlon_from_nc(files[step-1], "V", nlat, nlon, is_2D=True)
    data = spherical_vorticity_torch(data_u, data_v, 7E7, nlat, nlon, planetary_omega=None)
    vmin, vmax = torch.min(data[:]).item(), torch.max(data[:]).item()
    # Make color limits symmetric
    vmin, vmax = -max(abs(vmin), abs(vmax)), max(abs(vmin), abs(vmax))

    # Process up to 'step' number of files
    for i, file in tqdm(enumerate(files[:step])):
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        # Set polar projections for pole views
        ax2 = plt.subplot(132, projection='polar')
        ax3 = plt.subplot(133, projection='polar')

        # Get data from netCDF file
        data_u = latlon_from_nc(file, "U", nlat, nlon, is_2D=True)
        data_v = latlon_from_nc(file, "V", nlat, nlon, is_2D=True)
        data = spherical_vorticity_torch(data_u, data_v, 7E7, nlat, nlon, planetary_omega=None)
        
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # Create equatorial view
        im1 = ax1.imshow(data[0,:,:], origin='lower', aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Equatorial View (iter {i})')
        
        # Create north pole view (using polar projection)
        theta = np.linspace(0, 2*np.pi, nlon)
        r = np.linspace(0, 1, nlat//2)
        T, R = np.meshgrid(theta, r)
        im2 = ax2.pcolormesh(T, R, data[0,:nlat//2,:], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax2.set_title(f'North Pole View (iter {i})')
        
        # Create south pole view (using polar projection)
        im3 = ax3.pcolormesh(T, R, torch.flip(data[0,nlat//2:,:], dims=[0]), cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax3.set_title(f'South Pole View (iter {i})')
                
        # Adjust layout
        plt.tight_layout()
        
        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Append to frames
        frames.append(image)
    
        # Close the figure
        plt.close()
    
    # Save as GIF
    imageio.mimsave(output_name, frames, fps=2)
    print(f"Animation saved as {output_name}")


if __name__ == "__main__":
    folder = "/home/sc0657/scratch/jvort_2d/run_0214_midStr/"
    pattern = "*.out3.*.nc"
    
    # Example usage
    animator_from_folder(
        folder=folder,
        pattern=pattern,
        step=300,
        output_name="vort_animation_midStr.gif"
    )