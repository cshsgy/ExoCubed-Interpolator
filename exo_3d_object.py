import torch
import numpy as np
from exo_interpolator import latlon_from_nc

class Exo3DObject:
    def __init__(self, file_path, nlat, nlon):
        self.file_path = file_path
        self.nlat = nlat
        self.nlon = nlon
        self.data = self.load_data()
        self.pres_name = "pres"
        self.rho_name = "rho"
        self.u_name = "U"
        self.v_name = "V"
    
    def load_data(self):
        return latlon_from_nc(self.file_path, "rho", self.nlat, self.nlon, is_2D=True)
    
    def get_data(self, variable_name):
        return latlon_from_nc(self.file_path, variable_name, self.nlat, self.nlon, is_2D=True)