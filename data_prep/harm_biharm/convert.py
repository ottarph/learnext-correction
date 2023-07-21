
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

def convert_checkpoints_to_npy(checkpoints: Iterable[int], prefix: str, cb_print: int = 20) -> None:
    """
        Each CG2 function from learnext mesh takes 240kB to store on disk.
        For 2400 checkpoints, with both harmonic and biharmonic, saving all
        checkpoints as .npy-files should take about one GB.
    """

    from tools.loading import load_mesh_meshview
    from conf import mesh_file_loc, harmonic_file_loc, biharmonic_file_loc, harmonic_label, biharmonic_label

    _, fluid_mesh, _ = load_mesh_meshview(mesh_file_loc)

    V = df.VectorFunctionSpace(fluid_mesh, "CG", 2, 2)
    harmonic = df.Function(V)
    biharmonic = df.Function(V)

    harmonic_file = df.XDMFFile(harmonic_file_loc)
    biharmonic_file = df.XDMFFile(biharmonic_file_loc)
    for checkpoint in checkpoints:
        if checkpoint % cb_print == 0:
            print(f"{checkpoint=}")
        harmonic_file.read_checkpoint(harmonic, harmonic_label, checkpoint)
        biharmonic_file.read_checkpoint(biharmonic, biharmonic_label, checkpoint)

        harmonic_np = CG2_vector_to_array(harmonic)
        biharmonic_np = CG2_vector_to_array(biharmonic)

        np.save(prefix+f".harmonic.{checkpoint:04}.npy", harmonic_np)
        np.save(prefix+f".biharmonic.{checkpoint:04}.npy", biharmonic_np)

    return


if __name__ == "__main__":

    from timeit import default_timer as timer

    start = timer()

    convert_checkpoints_to_npy(range(2400+1), prefix="data_prep/harm_biharm/data_store/learnextCG2")

    end = timer()
    
    print((end - start)*1e0)

