from exo_3d_object import Exo3DVariable, Exo3DObject, VariableNames
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_xy(exo_3d: Exo3DObject, variable_name: str | VariableNames | None = None, var: Exo3DVariable | None = None) -> plt.Figure:
    assert variable_name is not None or var is not None, "Either variable_name or var must be provided"
    assert var is None or variable_name is None, "Only one of variable_name or var should be provided"

    if var is None:
        data = exo_3d.get_data(variable_name)
    else:
        data = var.get_data()
    plt.figure(figsize=(10, 5))
    plt.imshow(data.cpu().numpy(), extent=[-180, 180, -90, 90])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(label=var.units.__str__())
    return plt.gcf()

def plot_xz(exo_3d: Exo3DObject, variable_name: str | VariableNames | None = None, var: Exo3DVariable | None = None) -> plt.Figure:
    assert variable_name is not None or var is not None, "Either variable_name or var must be provided"
    assert var is None or variable_name is None, "Only one of variable_name or var should be provided"

    if var is None:
        data = exo_3d.get_data(variable_name)
    else:
        data = var.get_data()

    if exo_3d.n_pres_lyr > 0:
        is_pressure = True
    else:
        is_pressure = False

    plt.figure(figsize=(10, 5))
    if is_pressure:
        plt.imshow(torch.flip(data, dims=[0]).cpu().numpy(), extent=[-180, 180, 0, 1], aspect="auto")
    else:
        raise NotImplementedError("Plotting xz is only implemented for pressure data")
    plt.xlabel("Longitude")
    if is_pressure:
        plt.ylabel("Pressure")
    else:
        plt.ylabel("Height")
    plt.colorbar(label=var.units.__str__())
    return plt.gcf()
