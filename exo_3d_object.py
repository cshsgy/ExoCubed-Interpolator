from enum import Enum
import torch
import numpy as np
from exo_interpolator import latlon_from_nc, height_to_pres
import warnings

class VariableNames(Enum):
    RHO = "rho"
    PRES = "press"
    U = "U"
    V = "V"
    W = "W"
    TEMP = "temperature"

class Operations(Enum):
    TimeAverage = 1
    ZonalAverage = 2
    TimeSlice = 3
    VerticalSlice = 4
    MeridionalAverage = 5

class Units(dict):
    def __init__(self, kg: int = 0, m: int = 0, s: int = 0, K: int = 0):
        self["kg"] = kg
        self["m"] = m
        self["s"] = s
        self["K"] = K
    
    def repr_one(self, item:str):
        if self[item] == 0:
            return ""
        elif self[item] == 1:
            return item
        else:
            return f"{item}^{self[item]}"
    
    def __str__(self):
        kg_str = self.repr_one("kg")
        m_str = self.repr_one("m")
        s_str = self.repr_one("s")
        K_str = self.repr_one("K")
        return f"{kg_str} {m_str} {s_str} {K_str}"
    
    def mul(self, other: "Units"):
        return Units(kg=self["kg"] + other["kg"], m=self["m"] + other["m"], s=self["s"] + other["s"], K=self["K"] + other["K"])
    
    def div(self, other: "Units"):
        return Units(kg=self["kg"] - other["kg"], m=self["m"] - other["m"], s=self["s"] - other["s"], K=self["K"] - other["K"])

def get_units(variable_name: VariableNames | str):
    if variable_name == VariableNames.RHO:
        return Units(kg=1, m=-1)
    elif variable_name == VariableNames.PRES:
        return Units(kg=1, m=-1, s=-2)
    elif variable_name == VariableNames.U:
        return Units(m=1, s=-1)
    elif variable_name == VariableNames.V:
        return Units(m=1, s=-1)
    elif variable_name == VariableNames.W:
        return Units(m=1, s=-1)
    elif variable_name == VariableNames.TEMP:
        return Units(K=1)
    else:
        return None

class Exo3DVariable:
    def __init__(self, name: VariableNames | str, data: torch.Tensor, units: Units | None = None, operations: list[Operations] | None = None):
        self.name = name
        self.data = data
        self.units = units
        self.operations = operations if operations is not None else []
        
    def apply_operation(self, operation: Operations, slice_index: int | None = None):
        if operation in self.operations:
            warnings.warn(f"WARNING: Operation {operation} already applied to variable {self.name}, skipping")
            return
        time_dim = 0
        layer_dim = 1
        lat_dim = 2
        lon_dim = 3
        if Operations.TimeAverage in self.operations or Operations.TimeSlice in self.operations:
            layer_dim -= 1; lat_dim -= 1; lon_dim -= 1
        if Operations.VerticalSlice in self.operations:
            lat_dim -= 1; lon_dim -= 1
        if Operations.MeridionalAverage in self.operations:
            lon_dim -= 1
        if operation == Operations.TimeAverage:
            self.data = self.data.mean(dim=time_dim)
        elif operation == Operations.ZonalAverage:
            self.data = self.data.mean(dim=lon_dim)
        elif operation == Operations.TimeSlice:
            self.data = self.data[slice_index]
        elif operation == Operations.VerticalSlice:
            self.data = self.data[:, slice_index, :, :]
        elif operation == Operations.MeridionalAverage:
            self.data = self.data.mean(lat_dim)
        self.operations.append(operation)

    def get_shape(self):
        return self.data.shape

    def copy(self):
        return Exo3DVariable(self.name, self.data.clone(), self.units, self.operations.copy())
    
    def get_data(self, extended: bool = False):
        dims_to_extend = []
        if extended:
            if Operations.TimeAverage in self.operations or Operations.TimeSlice in self.operations:
                dims_to_extend.append(0)
            if Operations.VerticalSlice in self.operations:
                dims_to_extend.append(1)
            if Operations.MeridionalAverage in self.operations:
                dims_to_extend.append(2)
            if Operations.ZonalAverage in self.operations:
                dims_to_extend.append(3)

            if len(dims_to_extend) == 0:
                return self.data
        return_data = self.data.clone()
        for dim in dims_to_extend:
            return_data = return_data.unsqueeze(dim)
        return return_data

    def get_data_extended(self):
        return self.get_data(extended=True)

class Exo3DObject:
    def __init__(self, file_path: str, nlat: int, nlon: int, n_pres_lyr: int | None = None, time_index_range: tuple[int, int] | None = None):
        self.file_path = file_path
        self.nlat = nlat
        self.nlon = nlon
        self.shape4 = None
        self.variables: list[Exo3DVariable] = []
        if n_pres_lyr is not None:
            self.n_pres_lyr = n_pres_lyr
        else:
            self.n_pres_lyr = -1
        if time_index_range is not None:
            self.time_index_range = time_index_range
        else:
            self.time_index_range = (0, -1)
    
    def get_data(self, variable_name: VariableNames | str = VariableNames.RHO, units: Units | None = None):
        for variable in self.variables:
            if variable.name == variable_name:
                return variable.data
        data = latlon_from_nc(self.file_path, variable_name.value if isinstance(variable_name, VariableNames) else variable_name, self.nlat, self.nlon)
        data = data[self.time_index_range[0]:self.time_index_range[1], :, :, :]
        if self.shape4 is None:
            self.shape4 = data.shape
        if self.n_pres_lyr != -1 and variable_name != VariableNames.PRES:
            data, pres_shape4 = height_to_pres(data, self.get_data(VariableNames.PRES), n_pres_lyr=self.n_pres_lyr)
            self.shape4 = pres_shape4
        self.variables.append(Exo3DVariable(variable_name, data, units=units if units is not None else get_units(variable_name)))
        return data
    
    def get_variable(self, variable_name: VariableNames | str):
        for variable in self.variables:
            if variable.name == variable_name:
                return variable
        return None

    def get_eddy_flux(self, variable_name_1: VariableNames | str, variable_name_2: VariableNames | str, zonal_avg: bool = True, time_avg: bool = True) -> Exo3DVariable:
        '''
        Compute the eddy flux of two variables.
        Input variables must have the same shapes to calculate the residual fluxes. The returned variable will be the same shape as the input variables.
        The zonal_avg and time_avg options are used to determine the averaging method for the residual fluxes. If the variables already have the averaging applied, they will be ignored.
        '''
        assert zonal_avg or time_avg, "At least one of avg options must be True to compute residuals"
        var_1 = self.get_variable(variable_name_1)
        var_2 = self.get_variable(variable_name_2)
        assert var_1.get_shape() == var_2.get_shape(), "Variables must have the same shapes"
        var_1_avg = var_1.copy()
        var_2_avg = var_2.copy()
        if zonal_avg:
            var_1_avg.apply_operation(Operations.ZonalAverage)
            var_2_avg.apply_operation(Operations.ZonalAverage)
        if time_avg:
            var_1_avg.apply_operation(Operations.TimeAverage)
            var_2_avg.apply_operation(Operations.TimeAverage)
        var_1_residual = var_1_avg.get_data_extended() - var_1.get_data_extended()
        var_2_residual = var_2_avg.get_data_extended() - var_2.get_data_extended()
        residual_product = Exo3DVariable(name=f"{variable_name_1}_{variable_name_2}_eddy_flux", 
                                         data=torch.squeeze(var_1_residual * var_2_residual),
                                         units=var_1.units.mul(var_2.units),
                                         operations=var_1.operations.copy())
        residual_product.apply_operation(Operations.TimeAverage)
        residual_product.apply_operation(Operations.ZonalAverage)
        return residual_product
