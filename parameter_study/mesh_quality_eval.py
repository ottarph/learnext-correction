import dolfin as df
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import tqdm

from tools.mesh_quality import MeshQuality

from typing import Any
Mesh = Any

def compute_mesh_qualities(mesh: Mesh, dataloader: data.DataLoader, model: nn.Module,
                           quality_measure: str = "scaled_jacobian", 
                           show_progress_bar: bool = True) -> np.ndarray:
    """
    Compute the mesh quality of `mesh` deformed by prediction of `model` for inputs in `dataloader`.

    Args:
        mesh (df.Mesh): Mesh to deform and compute mesh quality over.
        dataloader (data.DataLoader): `DataLoader` of `torch.Tensor`s that are inputs to `model`.
        model (nn.Module): Neural network mesh motion model.
        quality_measure (str, optional): Mesh quality indicator to use. Defaults to "scaled_jacobian".
        show_progress_bar (bool, optional): Whether or not to show `tqdm` progress bar. Defaults to True.

    Returns:
        np.ndarray: `(num_steps, num_cells)`-shaped array containing mesh qualities for each cell and each input.
    """
    mesh_quality = MeshQuality(mesh, quality_measure=quality_measure)

    was_train_mode = model.training
    model.eval()

    num_steps = len(dataloader.dataset)
    mesh_quals = np.zeros((num_steps, mesh.num_cells()))
    k = 0

    if show_progress_bar:
        pbar = tqdm.tqdm(total=num_steps, desc="Computing mesh qualities")

    CG1 = df.VectorFunctionSpace(mesh, "CG", 1)
    u_cg1 = df.Function(CG1)
    scratch = np.zeros_like(u_cg1.vector()[:])
    for x, _ in dataloader:
        pred = model(x).detach().numpy()
        for i in range(pred.shape[0]):
            scratch[0::2] = pred[i,:,0]
            scratch[1::2] = pred[i,:,1]
            u_cg1.vector()[:] = scratch
            mesh_quals[k,:] = mesh_quality(u_cg1)
            k += 1

            if show_progress_bar:
                pbar.update()

    if was_train_mode:
        model.train()

    return mesh_quals
