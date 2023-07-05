
import dolfin as df
import numpy as np

from typing import Iterable

"""
    Convert a bunch of biharmonic CG2 checkpoints into a bunch of tensors that 
    correspond to the correct dof ordering.

    All of these functions are assuming serial running.

"""


def CG2_vector_to_array(u: df.Function) -> np.ndarray:

    raw_array = u.vector().get_local()

    return np.column_stack((raw_array[::2], raw_array[1::2]))

def convert_checkpoints_to_npy(checkpoints: Iterable[int], prefix: str) -> None:


    from tools.loading import load_mesh, load_harmonic_data, load_biharmonic_data
    from conf import OutputLoc

    _, fluid_mesh, _ = load_mesh(OutputLoc + "/Mesh_Generation")

    V = df.VectorFunctionSpace(fluid_mesh, "CG", 2, 2)
    harmonic = df.Function(V)
    biharmonic = df.Function(V)

    harmonic_file = df.XDMFFile(OutputLoc + "/Extension/Data/" + "input_.xdmf")
    biharmonic_file = df.XDMFFile(OutputLoc + "/Extension/Data/" + "output_.xdmf")
    for checkpoint in checkpoints:
        harmonic_file.read_checkpoint(harmonic, "input", checkpoint)
        biharmonic_file.read_checkpoint(biharmonic, "output", checkpoint)

        harmonic_np = CG2_vector_to_array(harmonic)
        biharmonic_np = CG2_vector_to_array(biharmonic)

        np.save(prefix+f".harmonic.{checkpoint:04}.npy", harmonic_np)
        np.save(prefix+f".biharmonic.{checkpoint:04}.npy", biharmonic_np)



    return


if __name__ == "__main__":


    convert_checkpoints_to_npy(range(10), prefix="data_prep/data_store/learnextCG2")

