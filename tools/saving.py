import torch
import torch.nn as nn
import numpy as np
import dolfin as df
from torch.utils.data import DataLoader
from tqdm import tqdm
import pathlib


def save_extensions_to_xdmf(model: nn.Module, dataloader: DataLoader, function_space: df.FunctionSpace, 
                            save_name, save_dir: str = "fenics_output", start_checkpoint: int = 0) -> None:
    
    if pathlib.Path(f"{save_dir}/{save_name}.xdmf").is_file():
        pathlib.Path(f"{save_dir}/{save_name}.xdmf").unlink()
    
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    was_train_mode = model.training
    model.eval()

    save_label = "predicted_extension"

    u_pred = df.Function(function_space)
    new_coeffs = np.zeros_like(u_pred.vector().get_local())
    k = start_checkpoint
    with df.XDMFFile(f"{save_dir}/{save_name}.xdmf") as outfile:
        with torch.no_grad():
            pred_loop = tqdm(dataloader, position=0, desc="Writing predictions to file")
            for x, _ in pred_loop:
                pred = model(x)
                for i in range(pred.shape[0]):
                    coeffs = pred[i,:,:]
                    new_coeffs[::2] = coeffs[:,0].detach().numpy()
                    new_coeffs[1::2] = coeffs[:,1].detach().numpy()
                    u_pred.vector().set_local(new_coeffs)
                    outfile.write_checkpoint(u_pred, save_label, float(k), append=True)
                    k += 1

    if was_train_mode:
        model.train()

    return
