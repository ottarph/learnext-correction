import torch
import torch.nn as nn
import numpy as np

NUM_VERTICES = 3935
BATCH_SIZE = 128

torch.set_default_dtype(torch.float32)
torch.manual_seed(seed=0)

if __name__ != "__main__":
    import pytest
    pytest.skip("These are failing based on which combination they are called in, very strange. "+
                "Also not implemented functionality at the time.", allow_module_level=True)

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

get_x = get_x_random
# get_x = get_x_dataset_notrans
# get_x = get_x_dataset_wtrans

def test_tensor_module():
    from networks.general import TensorModule

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mask_tensor = torch.load("models/mask_tensor_submesh.pt")
    tensor_module = TensorModule(mask_tensor)
    tensor_module_comp = torch.compile(tensor_module)
    tensor_module_comp.to(device)

    x = get_x().to(device)
    
    print(tensor_module_comp(x).shape)
    assert tensor_module_comp(x).shape == (NUM_VERTICES, )

    return

def test_prepend_module():
    from networks.general import PrependModule

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dof_coordinates = torch.load("models/dof_coordinates_submesh.pt")
    prepend_module = PrependModule(dof_coordinates)
    prepend_module_comp = torch.compile(prepend_module)
    prepend_module_comp.to(device)

    x = get_x().to(device)
    
    print(prepend_module_comp(x).shape)
    assert prepend_module_comp(x).shape == (BATCH_SIZE, NUM_VERTICES, 8)

    return

def test_prepend_mlp_stack():
    from networks.general import PrependModule, MLP

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dof_coordinates = torch.load("models/dof_coordinates_submesh.pt")
    prepend_module = PrependModule(dof_coordinates)
    mlp = MLP([8, 32, 2], activation=nn.ReLU())
    prepend_mlp_stack = nn.Sequential(prepend_module, mlp)
    prepend_mlp_stack_comp = torch.compile(prepend_mlp_stack)
    prepend_mlp_stack_comp.to(device)

    x = get_x().to(device)
    
    print(prepend_mlp_stack_comp(x).shape)
    assert prepend_mlp_stack_comp(x).shape == (BATCH_SIZE, NUM_VERTICES, 2)

    return

def test_trim_module():
    from networks.general import TrimModule

    device = "cuda" if torch.cuda.is_available() else "cpu"

    forward_indices = [range(2)]
    trim_module = TrimModule(forward_indices=forward_indices)
    trim_module_comp = torch.compile(trim_module)
    trim_module_comp.to(device)

    x = get_x().to(device)

    print(trim_module_comp(x).shape)
    assert trim_module_comp(x).shape == (BATCH_SIZE, NUM_VERTICES, 2)

    return

def test_masknet_stack():
    from networks.general import TensorModule, PrependModule, MLP, TrimModule
    from networks.masknet import MaskNet

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Base network
    forward_indices = [range(2)]
    trim_module = TrimModule(forward_indices=forward_indices)

    # Mask network
    mask_tensor = torch.load("models/mask_tensor_submesh.pt")
    tensor_module = TensorModule(mask_tensor)

    # Correction network
    dof_coordinates = torch.load("models/dof_coordinates_submesh.pt")
    prepend_module = PrependModule(dof_coordinates)
    mlp = MLP([8, 256, 2], activation=nn.ReLU())
    prepend_mlp_stack = nn.Sequential(prepend_module, mlp)

    # Masknet
    masknet = MaskNet(prepend_mlp_stack, trim_module, tensor_module)
    masknet_comp = torch.compile(masknet)
    masknet_comp.to(device)

    x = get_x().to(device)

    print(masknet_comp(x).shape)
    assert masknet_comp(x).shape == (BATCH_SIZE, NUM_VERTICES, 2)

    return

if __name__ == "__main__":
    print(f"{NUM_VERTICES = }")
    print(f"{BATCH_SIZE = }")

    # for get_x in [get_x_random, get_x_dataset_notrans, get_x_dataset_wtrans]:
    for get_x in [get_x_random]:
        # test_tensor_module()
        test_prepend_module()
        # test_prepend_mlp_stack()
        # test_trim_module()
        test_masknet_stack()

    """ 
        Crashes if only test_masknet_stack() is called,
        but runs fine if test_prepend_module() is called
        before test_masknet_stack(). Strange.
    """
