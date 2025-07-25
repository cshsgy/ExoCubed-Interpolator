from exo_3d_object import Exo3DObject, VariableNames, Operations, Units
from plotting_utils import plot_xy, plot_xz

if __name__ == "__main__":
    exo_3d = Exo3DObject(
        file_path="hotjupiter-main.nc", 
        nlat=50, 
        nlon=100, 
        n_pres_lyr=40, 
        time_index_range=(0, 2)
        )

    var_1 = exo_3d.get_data(VariableNames.RHO)
    var_2 = exo_3d.get_data("vel2", units=Units(m=1, s=-1))

    print([variable.name for variable in exo_3d.variables])

    eddy_flux = exo_3d.get_eddy_flux(VariableNames.RHO, "vel2")

    print(eddy_flux.get_data().shape)
    print(eddy_flux.name)

    figure = plot_xz(exo_3d, var=eddy_flux)
    figure.savefig("rho_vel2_eddy_flux.png", dpi=600)