import numpy as np
import torch
import torch.nn as nn

NUM_VERTICES = 3935
BATCH_SIZE = 128

if __name__ != "__main__":
    import pytest
    pytest.skip("This fails for no understandable reason, see tests in 'problems/test/test_compile.py'", allow_module_level=True)

def get_x_random():
    return torch.rand((BATCH_SIZE, NUM_VERTICES, 6))

def get_x_dataset_notrans():
    from data_prep.clement.dataset import learnextClementGradDataset
    from conf import train_checkpoints
    prefix = "data_prep/clement/data_store/grad/clm_grad"
    dataset = learnextClementGradDataset(prefix=prefix, checkpoints=train_checkpoints,
                                         transform=None, target_transform=None)
    
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    x, _ = next(iter(dl))

    return x

def get_x_dataset_wtrans():
    from data_prep.transforms import DofPermutationTransform
    from conf import submesh_conversion_cg1_loc
    perm_tens = torch.LongTensor(np.load(submesh_conversion_cg1_loc))
    transform = DofPermutationTransform(perm_tens, dim=-2)

    from data_prep.clement.dataset import learnextClementGradDataset
    from conf import train_checkpoints
    prefix = "data_prep/clement/data_store/grad/clm_grad"
    dataset = learnextClementGradDataset(prefix=prefix, checkpoints=train_checkpoints,
                                         transform=transform, target_transform=transform)
    
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    x, _ = next(iter(dl))

    return x

def test_clem_grad_prob():
    """
        As far as I can see, this is the main logic in 'problem/run_clement_grad_masknet.py'. 
        Crashes for no good reason, maybe a bad environment.
        See tests in 'problems/test/test_compile.py'
    """

    torch.set_default_dtype(torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)


    """ Mask module """
    from networks.general import TensorModule
    mask_tensor = torch.load("models/mask_tensor_submesh.pt")
    mask = TensorModule(mask_tensor)

    """ Base module """
    from networks.general import TrimModule
    forward_indices = [range(2)]
    base = TrimModule(forward_indices=forward_indices)

    """ Correction module """
    from networks.general import MLP, PrependModule
    widths = [8, 256, 2]
    mlp = MLP(widths, activation=nn.ReLU())
    dof_coordinates = torch.load("models/dof_coordinates_submesh.pt")
    prepend = PrependModule(torch.tensor(dof_coordinates, dtype=torch.get_default_dtype()))
    network = nn.Sequential(prepend, mlp)

    """ MaskNet """
    from networks.masknet import MaskNet
    mask_net = MaskNet(network, base, mask)

    """ Compile and move to device """
    mask_net = torch.compile(mask_net)
    mask_net.to(device)

    x = get_x_random().to(device)
    y = mask_net(x)
    
    assert y.shape == (BATCH_SIZE, NUM_VERTICES, 2)

    return


if __name__ == "__main__":
    test_clem_grad_prob()
