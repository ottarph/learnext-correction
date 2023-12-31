import dolfin as df
import numpy as np
from tqdm import tqdm

from typing import Iterable

from data_prep.clement.clement import clement_interpolate


def CG1_vector_to_array(u: df.Function) -> np.ndarray:
    """ 
        Layout: Columns ``(u_x, u_y)``
    """

    raw_array = u.vector().get_local()

    return np.column_stack((raw_array[0::2], raw_array[1::2]))

def CG1_vector_plus_grad_to_array(u: df.Function, du: df.Function) -> np.ndarray:
    """ 
        Layout: Columns ``(u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)``

        `u` is a CG1 function. `du` is a CG1 function over same 
        mesh as `u`, and a clement interpolant as given by 
            `clement_interpolate(df.grad(u))`,
    """

    new_array = np.zeros((u.function_space().mesh().num_vertices(), 6))
    
    raw_array_base = u.vector().get_local()
    new_array[:,0+0] = raw_array_base[0::2]
    new_array[:,0+1] = raw_array_base[1::2]

    raw_array_grad = du.vector().get_local()
    new_array[:,2+0] = raw_array_grad[0::4]
    new_array[:,2+1] = raw_array_grad[1::4]
    new_array[:,2+2] = raw_array_grad[2::4]
    new_array[:,2+3] = raw_array_grad[3::4]

    return new_array

def convert_checkpoints_to_npy_clement_grad(checkpoints: Iterable[int], fname_harm: str, fname_biharm: str, prefix: str,
                                            save_offset: int = 0) -> None:
    """
        Takes CG1 functions and saves them as np.ndarray along with their
        Clement interpolant of the gradient.
    """

    fluid_mesh_loc = "data_prep/artificial/working_space/fluid.h5"
    fluid_mesh = df.Mesh()
    with df.HDF5File(fluid_mesh.mpi_comm(), fluid_mesh_loc, 'r') as h5:
        h5.read(fluid_mesh, 'mesh', False)

    harmonic_label = "u_harm_cg1"
    biharmonic_label = "u_biharm_cg1"

    V_cg1 = df.VectorFunctionSpace(fluid_mesh, "CG", 1, 2)
    harmonic_cg1 = df.Function(V_cg1)
    biharmonic_cg1 = df.Function(V_cg1)

    _, clement_interpolater_harm = clement_interpolate(df.grad(harmonic_cg1), with_CI=True)

    harmonic_file = df.XDMFFile(fname_harm)
    biharmonic_file = df.XDMFFile(fname_biharm)
    for checkpoint in tqdm(checkpoints):

        harmonic_file.read_checkpoint(harmonic_cg1, harmonic_label, checkpoint)
        biharmonic_file.read_checkpoint(biharmonic_cg1, biharmonic_label, checkpoint)

        gh_harm = clement_interpolater_harm()

        harmonic_np = CG1_vector_plus_grad_to_array(harmonic_cg1, gh_harm)
        biharmonic_np = CG1_vector_to_array(biharmonic_cg1)

        np.save(prefix+f".harm_plus_clm_grad.{checkpoint+save_offset:04}.npy", harmonic_np)
        np.save(prefix+f".biharm.{checkpoint+save_offset:04}.npy", biharmonic_np)

    return


if __name__ == "__main__":

    working_dir = "data_prep/artificial/working_space"

    checkpoints1 = range(101)
    checkpoints2 = range(101)
    checkpoints3 = range(101)
    checkpoints4 = range(101)
    checkpoints5 = range(101)
    checkpoints6 = range(101)
    convert_checkpoints_to_npy_clement_grad(checkpoints1, working_dir+"/harmonic1.xdmf", working_dir+"/biharmonic1.xdmf",
                                            prefix="data_prep/artificial/data_store/grad/art", save_offset=0)
    convert_checkpoints_to_npy_clement_grad(checkpoints1, working_dir+"/harmonic2.xdmf", working_dir+"/biharmonic2.xdmf",
                                            prefix="data_prep/artificial/data_store/grad/art", save_offset=101)
    convert_checkpoints_to_npy_clement_grad(checkpoints1, working_dir+"/harmonic3.xdmf", working_dir+"/biharmonic3.xdmf",
                                            prefix="data_prep/artificial/data_store/grad/art", save_offset=202)
    convert_checkpoints_to_npy_clement_grad(checkpoints1, working_dir+"/harmonic4.xdmf", working_dir+"/biharmonic4.xdmf",
                                            prefix="data_prep/artificial/data_store/grad/art", save_offset=303)
    convert_checkpoints_to_npy_clement_grad(checkpoints1, working_dir+"/harmonic5.xdmf", working_dir+"/biharmonic5.xdmf",
                                            prefix="data_prep/artificial/data_store/grad/art", save_offset=404)
    convert_checkpoints_to_npy_clement_grad(checkpoints1, working_dir+"/harmonic6.xdmf", working_dir+"/biharmonic6.xdmf",
                                            prefix="data_prep/artificial/data_store/grad/art", save_offset=505)
    
