import torch
import torch.nn as nn
import dolfin as df
import numpy as np

from torch.utils.data import DataLoader

from timeit import default_timer as timer

def main():
    torch.set_default_dtype(torch.float32)
    device = "cpu"
    print(f"{device = }")

    """ Select problem setup """
    run_name = "golf"
    model_dir = f"models/artificial/{run_name}"


    from conf import mesh_file_loc, with_submesh
    from tools.loading import load_mesh
    fluid_mesh = load_mesh(mesh_file_loc, with_submesh)

    V_scal = df.FunctionSpace(fluid_mesh, "CG", 1) # Linear scalar polynomials over triangular mesh

    from data_prep.transforms import DofPermutationTransform
    from conf import submesh_conversion_cg1_loc
    perm_tens = torch.LongTensor(np.load(submesh_conversion_cg1_loc))
    dof_perm_transform = DofPermutationTransform(perm_tens, dim=-2)
    transform = dof_perm_transform if with_submesh else None
    print(f"{with_submesh = }")

    from data_prep.clement.dataset import learnextClementGradDataset
    from conf import test_checkpoints
    # test_checkpoints = range(2401)
    prefix = "data_prep/clement/data_store/grad/clm_grad"
    test_dataset = learnextClementGradDataset(prefix=prefix, checkpoints=test_checkpoints,
                                         transform=transform, target_transform=transform)
    
    batch_size = 128
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    """ Build model """

    """ Base """

    from networks.general import TrimModule
    indices = torch.LongTensor(range(2))
    base = TrimModule(indices, dim=-1)

    """ Mask """

    V_scal = df.FunctionSpace(fluid_mesh, "CG", 1)
    from conf import poisson_mask_f
    from networks.masknet import poisson_mask_custom
    mask_df = poisson_mask_custom(V_scal, poisson_mask_f, normalize=True)

    from networks.general import TensorModule
    mask = TensorModule(torch.tensor(mask_df.vector().get_local(), dtype=torch.get_default_dtype()))

    """ Correction """
    coordinates = V_scal.tabulate_dof_coordinates()

    from networks.general import PrependModule
    prepend = PrependModule(torch.tensor(coordinates, dtype=torch.get_default_dtype()))

    from networks.loading import load_model
    norm_mlp_stack = load_model(model_dir = model_dir, load_state_dict = True, mode = "yaml")
    
    correction = nn.Sequential(prepend, norm_mlp_stack)

    from networks.masknet import MaskNet
    mask_net = MaskNet(correction, base, mask)
    mask_net.to(device)
    mask_net.eval()


    xdmf_dir = "fenics_output/artificial"
    xdmf_name = f"pred_test_artificial_{run_name}"
    from tools.saving import save_extensions_to_xdmf
    mask_net.to("cpu")
    save_extensions_to_xdmf(mask_net, test_dataloader, df.VectorFunctionSpace(fluid_mesh, "CG", 1), xdmf_name,
                            save_dir=xdmf_dir, start_checkpoint=test_checkpoints[0])

if __name__ == "__main__":
    main()
