
import dolfin as df
import numpy as np

from data_prep.clement.clement import *

from typing import Iterable

"""
    Convert a bunch of biharmonic CG2 checkpoints into a bunch of tensors that 
    correspond to the correct CG1 dof ordering, but include a clement interpolant
    of gradient (and hessian).

    All of these functions are assuming serial running.

"""



def CG1_vector_to_array(u: df.Function) -> np.ndarray:
    """ 
        Layout: Columns ``(u_x, u_y)``
    """

    raw_array = u.vector().get_local()

    return np.column_stack((raw_array[::2], raw_array[1::2]))

def CG1_vector_plus_grad_to_array(u: df.Function, du: df.Function) -> np.ndarray:
    """ 
        Layout: Columns ``(u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)``

        `u` is a CG1 function. `du` is a CG1 function over same 
        mesh as `u`, and a clement interpolant as given by 
            `clement_interpolate(df.grad(qh))`,
        where `qh` is taken as
            `df.interpolate(u, df.VectorFunctionSpace(u.function_space().mesh(), "DG", 1, 2))`.
    """

    new_array = np.zeros((u.function_space().mesh().num_vertices(), 6))
    
    raw_array_base = u.vector().get_local()
    new_array[:,0] = raw_array_base[::2]
    new_array[:,1] = raw_array_base[1::2]

    raw_array_grad = du.vector().get_local()
    new_array[:,2+0] = raw_array_grad[0::4]
    new_array[:,2+1] = raw_array_grad[1::4]
    new_array[:,2+2] = raw_array_grad[2::4]
    new_array[:,2+3] = raw_array_grad[3::4]

    return new_array

def convert_checkpoints_to_npy_clement_grad(checkpoints: Iterable[int], prefix: str, cb_print: int = 20) -> None:
    """
        Takes CG2 functions and saves as np.ndarray their CG1 projection along with their
        Clement interpolant of the gradient.
    """
    
    from tools.loading import load_mesh
    from conf import OutputLoc

    _, fluid_mesh, _ = load_mesh(OutputLoc + "/Mesh_Generation")

    V_cg2 = df.VectorFunctionSpace(fluid_mesh, "CG", 2, 2)
    harmonic_cg2 = df.Function(V_cg2)
    biharmonic_cg2 = df.Function(V_cg2)

    V_cg1 = df.VectorFunctionSpace(fluid_mesh, "CG", 1, 2)
    harmonic_cg1 = df.Function(V_cg1)
    biharmonic_cg1 = df.Function(V_cg1)

    Q = df.VectorFunctionSpace(fluid_mesh, "DG", 1, 2)


    harmonic_file = df.XDMFFile(OutputLoc + "/Extension/Data/" + "input_.xdmf")
    biharmonic_file = df.XDMFFile(OutputLoc + "/Extension/Data/" + "output_.xdmf")
    for checkpoint in checkpoints:
        if checkpoint % cb_print == 0:
            print(f"{checkpoint=}")

        harmonic_file.read_checkpoint(harmonic_cg2, "input", checkpoint)
        biharmonic_file.read_checkpoint(biharmonic_cg2, "output", checkpoint)

        harmonic_cg1 = df.interpolate(harmonic_cg2, V_cg1)
        biharmonic_cg1 = df.interpolate(biharmonic_cg2, V_cg1)

        qh_harm = df.interpolate(harmonic_cg1, Q)
        gh_harm = clement_interpolate(df.grad(qh_harm))

        harmonic_np = CG1_vector_plus_grad_to_array(harmonic_cg1, gh_harm)
        biharmonic_np = CG1_vector_to_array(biharmonic_cg1)

        np.save(prefix+f".harm_plus_clm_grad.{checkpoint:04}.npy", harmonic_np)
        np.save(prefix+f".biharm.{checkpoint:04}.npy", biharmonic_np)

    return



if __name__ == "__main__":

    from timeit import default_timer as timer

    start = timer()

    convert_checkpoints_to_npy_clement_grad(range(2400+1), prefix="data_prep/clement/data_store/clm_grad", cb_print=20)

    end = timer()
    
    print((end - start)*1e0)

