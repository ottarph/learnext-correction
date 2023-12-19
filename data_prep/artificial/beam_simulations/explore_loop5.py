import dolfin as df
import numpy as np
from pathlib import Path

from data_prep.artificial.elasticity import solve_neohook_solid, solve_linear_solid


if __name__ == "__main__":

    working_dir = "data_prep/artificial/working_space"

    solid_mesh = df.Mesh()
    with df.HDF5File(solid_mesh.mpi_comm(), working_dir+"/solid.h5", 'r') as h5:
        h5.read(solid_mesh, 'mesh', False)

    tdim = solid_mesh.topology().dim()
    solid_boundaries = df.MeshFunction('size_t', solid_mesh, tdim-1, 0)
    with df.HDF5File(solid_mesh.mpi_comm(), working_dir+"/solid.h5", 'r') as h5:
        h5.read(solid_boundaries, 'boundaries')

    p = Path(working_dir+"/displacements5.xdmf")
    if p.exists():
        p.unlink()
    p = Path(working_dir+"/displacements5.h5")
    if p.exists():
        p.unlink()
    displacement_file = df.XDMFFile(working_dir+"/displacements5.xdmf")

    # ----6----
    # 4       9
    # ----6----

    for k, theta in enumerate(np.linspace(0, 2*np.pi, 101)):
        displacement_bcs = {4: df.Constant((0, 0))}
        volume_load = df.Expression(('0', '-A*x[0]'), degree=1, A=1E-3*10*-7.2*0)

        six_load = df.Expression(('0', 'abs(x[0] - 0.45) < 0.04? C: 0'), degree=1, C=-0.66E3*np.cos(theta+0.0*np.pi)*0)
        nine_load = df.Constant((0, 0.53E3*np.cos(theta)))
        surface_load = {9: nine_load, 6: six_load}

        solvers = {"neohook": solve_neohook_solid, "linear": solve_linear_solid}
        solid_type = "neohook"
        # solid_type = "linear"

        uh = solvers[solid_type](boundaries=solid_boundaries,
                                volume_load=volume_load,
                                surface_load=surface_load,
                                displacement_bcs=displacement_bcs,
                                mu=df.Constant(0.5e6),
                                lmbda=df.Constant(2.0e6), pdegree=2)

        displacement_file.write_checkpoint(uh, "uh", k, append=True)

    displacement_file.close()

   