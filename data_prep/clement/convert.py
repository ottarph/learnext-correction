
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

def CG1_vector_plus_grad_hess_to_array(u: df.Function, du: df.Function, 
                                       Hu_1: df.Function, Hu_2: df.Function) -> np.ndarray:
    """ 
        Layout: Columns 
                        (u_x, u_y, 
                        d_x u_x,  d_y u_x,  d_x u_y,  d_y u_y,
                       d_xx u_x, d_xy u_x, d_yx u_x, d_yy u_x,
                       d_xx u_y, d_xy u_y, d_yx u_y, d_yy u_y)
        
    """

    new_array = np.zeros((u.function_space().mesh().num_vertices(), 14))
    
    raw_array_base = u.vector().get_local()
    new_array[:,0] = raw_array_base[::2]
    new_array[:,1] = raw_array_base[1::2]

    raw_array_grad = du.vector().get_local()
    new_array[:,2+0] = raw_array_grad[0::4]
    new_array[:,2+1] = raw_array_grad[1::4]
    new_array[:,2+2] = raw_array_grad[2::4]
    new_array[:,2+3] = raw_array_grad[3::4]

    raw_array_hess_1 = Hu_1.vector().get_local()
    new_array[:,6+0] = raw_array_hess_1[0::4]
    new_array[:,6+1] = raw_array_hess_1[1::4]
    new_array[:,6+2] = raw_array_hess_1[2::4]
    new_array[:,6+3] = raw_array_hess_1[3::4]

    raw_array_hess_2 = Hu_2.vector().get_local()
    new_array[:,10+0] = raw_array_hess_2[0::4]
    new_array[:,10+1] = raw_array_hess_2[1::4]
    new_array[:,10+2] = raw_array_hess_2[2::4]
    new_array[:,10+3] = raw_array_hess_2[3::4]

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


def convert_checkpoints_to_npy_clement_grad_hess(checkpoints: Iterable[int], prefix: str, cb_print: int = 20) -> None:
    """
        Takes CG2 functions and saves as np.ndarray their CG1 projection along with their
        Clement interpolant of the gradient and hessian.

        Data structure:
            harmonic: (u_x, u_y, 
                        d_x u_x,  d_y u_x,  d_x u_y,  d_y u_y,
                       d_xx u_x, d_xy u_x, d_yx u_x, d_yy u_x,
                       d_xx u_y, d_xy u_y, d_yx u_y, d_yy u_y)

            biharmonic: (u_x, u_y)

        num_vertices * (2 + 4 + 4 + 4) = 3935 * 14 = 55090 
        double precision floats per checkpoint. 2401 checkpoints
        results in about 1 gigabyte data.
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

    Q_grad = df.VectorFunctionSpace(fluid_mesh, "DG", 1, 2)
    Q_hess = df.VectorFunctionSpace(fluid_mesh, "DG", 2, 2)


    harmonic_file = df.XDMFFile(OutputLoc + "/Extension/Data/" + "input_.xdmf")
    biharmonic_file = df.XDMFFile(OutputLoc + "/Extension/Data/" + "output_.xdmf")
    for checkpoint in checkpoints:
        if checkpoint % cb_print == 0:
            print(f"{checkpoint=}")

        harmonic_file.read_checkpoint(harmonic_cg2, "input", checkpoint)
        biharmonic_file.read_checkpoint(biharmonic_cg2, "output", checkpoint)

        harmonic_cg1 = df.interpolate(harmonic_cg2, V_cg1)
        biharmonic_cg1 = df.interpolate(biharmonic_cg2, V_cg1)

        qh_dg1 = df.interpolate(harmonic_cg1, Q_grad)
        qh_dg2 = df.interpolate(harmonic_cg2, Q_hess)
        qh_dg2_1, qh_dg2_2 = qh_dg2.split()

        grad_h = clement_interpolate(df.grad(qh_dg1))
        hess_h_1 = clement_interpolate(df.grad(df.grad(qh_dg2_1)))
        hess_h_2 = clement_interpolate(df.grad(df.grad(qh_dg2_2)))

        harmonic_np = CG1_vector_plus_grad_hess_to_array(harmonic_cg1, grad_h, hess_h_1, hess_h_2)
        biharmonic_np = CG1_vector_to_array(biharmonic_cg1)

        np.save(prefix+f".harm_plus_clm_grad_hess.{checkpoint:04}.npy", harmonic_np)
        np.save(prefix+f".biharm.{checkpoint:04}.npy", biharmonic_np)

    return



if __name__ == "__main__":

    from timeit import default_timer as timer

    start = timer()

    # convert_checkpoints_to_npy_clement_grad(range(2400+1), 
    #           prefix="data_prep/clement/data_store/grad/clm_grad", cb_print=20)
    # convert_checkpoints_to_npy_clement_grad_hess(range(2400+1), 
    #             prefix="data_prep/clement/data_store/grad_hess/clm_grad_hess", cb_print=20)

    end = timer()
    
    print((end - start)*1e0)

