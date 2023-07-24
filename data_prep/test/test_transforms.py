from data_prep.transforms import *


def test_dof_permutation():

    from conf import submesh_conversion_cg1_loc, submesh_conversion_cg2_loc

    perm_tens = torch.tensor(np.load(submesh_conversion_cg1_loc))
    transform = DofPermutationTransform(perm_tens, -2)

    x = torch.rand(2, perm_tens.shape[0], 2)
    y = transform(x)
    assert torch.equal(x[:, perm_tens, :], y)

    perm_tens_2 = torch.tensor(np.load(submesh_conversion_cg2_loc))
    transform_2 = DofPermutationTransform(perm_tens_2)

    x = torch.rand(4, 3, perm_tens_2.shape[0], 2)
    y = transform_2(x)
    assert torch.equal(x[..., perm_tens_2, :], y)

    return

def test_dataset_with_dof_perm_transform_harm_biharm():

    from conf import submesh_conversion_cg2_loc, submesh_conversion_cg1_loc
    from data_prep.harm_biharm.dataset import learnextDataset
    from torch.utils.data import DataLoader

    """ ----------------------CG2---------------------------- """
    prefix = "data_prep/harm_biharm/data_store/learnextCG2"
    checkpoints = range(0, 2400+1)

    perm_tens = torch.tensor(np.load(submesh_conversion_cg2_loc))
    transform = DofPermutationTransform(perm_tens, dim=-2)

    dataset0 = learnextDataset(prefix, checkpoints, 
                              transform=None, target_transform=None)
    dataset = learnextDataset(prefix, checkpoints, 
                              transform=transform, target_transform=transform)
    

    dataloader0 = DataLoader(dataset0)
    dataloader = DataLoader(dataset)

    dl_iter_0 = iter(dataloader0)
    dl_iter = iter(dataloader)

    x0, y0 = next(dl_iter_0)
    x, y = next(dl_iter)

    assert torch.equal(x0[:, perm_tens, :], x)
    assert torch.equal(y0[:, perm_tens, :], y)

    return

def test_dataset_with_dof_perm_transform_clm_grad():

    from conf import submesh_conversion_cg1_loc
    from data_prep.clement.dataset import learnextClementGradDataset
    from torch.utils.data import DataLoader

    prefix = "data_prep/clement/data_store/grad/clm_grad"
    checkpoints = range(0, 2400+1)

    perm_tens = torch.tensor(np.load(submesh_conversion_cg1_loc))
    transform = DofPermutationTransform(perm_tens, dim=-2)

    dataset0 = learnextClementGradDataset(prefix, checkpoints, 
                              transform=None, target_transform=None)
    dataset = learnextClementGradDataset(prefix, checkpoints, 
                              transform=transform, target_transform=transform)
    

    dataloader0 = DataLoader(dataset0)
    dataloader = DataLoader(dataset)

    dl_iter_0 = iter(dataloader0)
    dl_iter = iter(dataloader)

    x0, y0 = next(dl_iter_0)
    x, y = next(dl_iter)

    assert torch.equal(x0[:, perm_tens, :], x)
    assert torch.equal(y0[:, perm_tens, :], y)

    return

def test_dataset_with_dof_perm_transform_clm_grad_hess():

    from conf import submesh_conversion_cg1_loc
    from data_prep.clement.dataset import learnextClementGradHessDataset
    from torch.utils.data import DataLoader

    prefix = "data_prep/clement/data_store/grad_hess/clm_grad_hess"
    checkpoints = range(0, 2400+1)

    perm_tens = torch.tensor(np.load(submesh_conversion_cg1_loc))
    transform = DofPermutationTransform(perm_tens, dim=-2)

    dataset0 = learnextClementGradHessDataset(prefix, checkpoints, 
                              transform=None, target_transform=None)
    dataset = learnextClementGradHessDataset(prefix, checkpoints, 
                              transform=transform, target_transform=transform)
    

    dataloader0 = DataLoader(dataset0)
    dataloader = DataLoader(dataset)

    dl_iter_0 = iter(dataloader0)
    dl_iter = iter(dataloader)

    x0, y0 = next(dl_iter_0)
    x, y = next(dl_iter)

    assert torch.equal(x0[:, perm_tens, :], x)
    assert torch.equal(y0[:, perm_tens, :], y)

    return


if __name__ == "__main__":
    test_dof_permutation()
    test_dataset_with_dof_perm_transform_harm_biharm()
    test_dataset_with_dof_perm_transform_clm_grad()
    test_dataset_with_dof_perm_transform_clm_grad_hess()
