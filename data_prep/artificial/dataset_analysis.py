import dolfin as df
import numpy as np
from tqdm import tqdm

""" Gather data to make some histograms over how the artificial dataset compares to the FSI dataset """


""" Data to gather: 
        - y-displacement
        - x-displacement
        - Jacobian of extension
        
    Analyse these for both harmonic and biharmonic extensions. """

dat_scheme = ("u_x", "u_y", "D_x u_x", "D_y u_x", "D_x u_y", "D_y u_y")

""" Sensor locations to measure pointwise data
    place these at the corners and midpoint of the tip of the flag,
    as well as on the midpoints along the side. """

sensors = [
    np.array([0.6, 0.19]),      # Lower corner of tip of flag
    np.array([0.6, 0.21]),      # Upper corner of tip of flag
    np.array([0.6, 0.20]),      # Midpoint of tip of flag
    np.array([0.424494897425, 0.19]),    # Approximate midpoint of lower side of flag
    np.array([0.424494897425, 0.21])     # Approximate midpoint of upper side of flag
]


""" Load meshes """

working_dir = "data_prep/artificial/working_space"

fluid_mesh = df.Mesh()
with df.HDF5File(fluid_mesh.mpi_comm(), working_dir+'/fluid.h5', 'r') as h5:
    h5.read(fluid_mesh, 'mesh', False)


""" Define function space and function to load checkpoints """
V = df.VectorFunctionSpace(fluid_mesh, "CG", 1)
u = df.Function(V)
u.set_allow_extrapolation(True)

""" Create a Clement interpolater to reconstruct Jacobian """
from data_prep.clement.clement import clement_interpolate
_, CI = clement_interpolate(df.grad(u), with_CI=True)


""" Gather artificial harmonic extension data """ 

""" Load a dataset """
datasets = [f"{working_dir}/harmonic{k}.xdmf" for k in range(1,6+1)]
checkpoint_ranges = [range(101) for _ in datasets]

dataset = datasets[0]
checkpoints = checkpoint_ranges[0]
infile = df.XDMFFile(dataset)


""" Create data array """
data = np.zeros((sum(map(len, checkpoint_ranges)), len(sensors), len(dat_scheme)))


""" Load a checkpoint """
infile.read_checkpoint(u, "u_harm_cg1", checkpoints[0])
Du = CI()
u(sensors[0])
Du(sensors[0])


""" Iterate through all datasets and save data """
for i in tqdm(range(len(datasets))):
    dataset = datasets[i]
    checkpoints = checkpoint_ranges[i]
    infile = df.XDMFFile(dataset)
    for k in tqdm(checkpoints, leave=False, position=1):
        infile.read_checkpoint(u, "u_harm_cg1", k)
        Du = CI()
        for j, p in enumerate(sensors):
            data[sum(map(len, checkpoint_ranges[:i]))+k, j, 0:2] = u(p)
            data[sum(map(len, checkpoint_ranges[:i]))+k, j, 2:6] = Du(p)


np.save("artificial_harm_datafile.npy", data)

print()
""" Gather artificial biharmonic extension data """

""" Load a dataset """
datasets = [f"{working_dir}/biharmonic{k}.xdmf" for k in range(1,6+1)]
checkpoint_ranges = [range(101) for _ in datasets]

dataset = datasets[0]
checkpoints = checkpoint_ranges[0]
infile = df.XDMFFile(dataset)


""" Create data array """
data = np.zeros((sum(map(len, checkpoint_ranges)), len(sensors), len(dat_scheme)))


""" Load a checkpoint """
infile.read_checkpoint(u, "u_biharm_cg1", checkpoints[0])
Du = CI()
u(sensors[0])
Du(sensors[0])


""" Iterate through all datasets and save data """
for i in tqdm(range(len(datasets))):
    dataset = datasets[i]
    checkpoints = checkpoint_ranges[i]
    infile = df.XDMFFile(dataset)
    for k in tqdm(checkpoints, leave=False, position=1):
        infile.read_checkpoint(u, "u_biharm_cg1", k)
        Du = CI()
        for j, p in enumerate(sensors):
            data[sum(map(len, checkpoint_ranges[:i]))+k, j, 0:2] = u(p)
            data[sum(map(len, checkpoint_ranges[:i]))+k, j, 2:6] = Du(p)


np.save("artificial_biharm_datafile.npy", data)

print()

""" Do the same for FSI dataset"""

""" Load meshes """

from conf import mesh_file_loc
from tools.loading import load_mesh

fluid_mesh = load_mesh(mesh_file_loc, with_submesh=False)




""" Define function space and function to load checkpoints """
V = df.VectorFunctionSpace(fluid_mesh, "CG", 2)
u = df.Function(V)
u.set_allow_extrapolation(True)

V_scal = df.FunctionSpace(fluid_mesh, "CG", 2)
dummy_bc = df.DirichletBC(V_scal, df.Constant(1.0), "on_boundary")
u_dummy = df.Function(V_scal)
dummy_bc.apply(u_dummy.vector())
boundary_dofs = np.argwhere(u_dummy.vector()).flatten()
sens_dofs = []
for sensor in sensors:
    ind = np.argmin(np.linalg.norm(V_scal.tabulate_dof_coordinates()[boundary_dofs] - sensor, axis=1))
    sens_dofs.append(boundary_dofs[ind])

sens_dofs = np.array(sens_dofs, dtype=int)


""" Create a Clement interpolater to reconstruct Jacobian """
from data_prep.clement.clement import clement_interpolate
_, CI = clement_interpolate(df.grad(u), with_CI=True)


""" Gather FSI harmonic extension data """ 
from conf import harmonic_file_loc, harmonic_label
""" Load a dataset """
datasets = [harmonic_file_loc]
checkpoint_ranges = [range(2400+1)]
# checkpoint_ranges = [range(120+1)]

dataset = datasets[0]
checkpoints = checkpoint_ranges[0]
infile = df.XDMFFile(dataset)


""" Create data array """
data = np.zeros((sum(map(len, checkpoint_ranges)), len(sensors), len(dat_scheme)))


""" Load a checkpoint """
infile.read_checkpoint(u, harmonic_label, checkpoints[0])
Du = CI()
u(sensors[0])
Du(sensors[0])



""" Iterate through all datasets and save data """
for i in tqdm(range(len(datasets))):
    dataset = datasets[i]
    checkpoints = checkpoint_ranges[i]
    infile = df.XDMFFile(dataset)
    for k in tqdm(checkpoints, leave=False, position=1):
        infile.read_checkpoint(u, harmonic_label, k)
        Du = CI()
        Du.set_allow_extrapolation(True)
        for j, p in enumerate(sensors):
            # data[sum(map(len, checkpoint_ranges[:i]))+k, j, 0:2] = u(p)
            data[sum(map(len, checkpoint_ranges[:i]))+k, j, 2:6] = Du(p)
            data[sum(map(len, checkpoint_ranges[:i]))+k, j, 0:2] = u.vector().get_local()[sens_dofs[j]*2:2*(sens_dofs[j]+1)]
            # data[sum(map(len, checkpoint_ranges[:i]))+k, j, 2:6] = Du.vector().get_local()[sens_dofs[j]*2:2*(sens_dofs[j]+1)]
            

np.save("fsi_harm_datafile.npy", data)

print()
""" Gather FSI biharmonic extension data """

from conf import biharmonic_file_loc, biharmonic_label
""" Load a dataset """
datasets = [biharmonic_file_loc]
checkpoint_ranges = [range(2400+1)]
# checkpoint_ranges = [range(120+1)]

dataset = datasets[0]
checkpoints = checkpoint_ranges[0]
infile = df.XDMFFile(dataset)


""" Create data array """
data = np.zeros((sum(map(len, checkpoint_ranges)), len(sensors), len(dat_scheme)))


""" Load a checkpoint """
infile.read_checkpoint(u, biharmonic_label, checkpoints[0])
Du = CI()
u(sensors[0])
Du(sensors[0])


""" Iterate through all datasets and save data """
for i in tqdm(range(len(datasets))):
    dataset = datasets[i]
    checkpoints = checkpoint_ranges[i]
    infile = df.XDMFFile(dataset)
    for k in tqdm(checkpoints, leave=False, position=1):
        infile.read_checkpoint(u, biharmonic_label, k)
        Du = CI()
        Du.set_allow_extrapolation(True)
        for j, p in enumerate(sensors):
            # data[sum(map(len, checkpoint_ranges[:i]))+k, j, 0:2] = u(p)
            data[sum(map(len, checkpoint_ranges[:i]))+k, j, 2:6] = Du(p)
            data[sum(map(len, checkpoint_ranges[:i]))+k, j, 0:2] = u.vector().get_local()[sens_dofs[j]*2:2*(sens_dofs[j]+1)]

np.save("fsi_biharm_datafile.npy", data)

