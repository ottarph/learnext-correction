from data_prep.clement.dataset import *

def test_clm_grad_dataset():
    prefix = "data_prep/clement/data_store/grad/clm_grad"
    checkpoints = range(0, 2400+1)

    dataset = learnextClementGradDataset(prefix, checkpoints)

    from torch.utils.data import DataLoader

    # torch.manual_seed(0)
    batch_size = 8
    shuffle = False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    harms, biharms = next(iter(dataloader))

    assert len(harms.shape) == 3
    assert harms.shape[-1] == 6 # u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y
    assert len(biharms.shape) == 3
    assert biharms.shape[-1] == 2 # u_x, u_y
    assert harms.shape[1] == 3935 # Dofs located at each of the 3935 vertices of the learnExt mesh.
    assert torch.linalg.norm(harms - harms[0,...]) > 0.0
    assert torch.linalg.norm(biharms - biharms[0,...]) > 0.0
    assert torch.all( torch.linalg.norm(biharms - harms[:,:,:2], dim=(1,2)) > 0.0 )

    try:
        x, y = dataset[checkpoints[0]]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    except :
        raise AssertionError
    try:
        x, y = dataset[checkpoints[1]]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    except :
        raise AssertionError
    try:
        x, y = dataset[checkpoints[-1]]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    except :
        raise AssertionError

    return
    
def test_clm_grad_hess_dataset():
    prefix = "data_prep/clement/data_store/grad_hess/clm_grad_hess"
    checkpoints = range(0, 2400+1)

    dataset = learnextClementGradHessDataset(prefix, checkpoints)

    from torch.utils.data import DataLoader

    # torch.manual_seed(0)
    batch_size = 8
    shuffle = False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    harms, biharms = next(iter(dataloader))

    #   (    u_x,      u_y, 
    #     d_x u_x,  d_y u_x,  d_x u_y,  d_y u_y,
    #    d_xx u_x, d_xy u_x, d_yx u_x, d_yy u_x,
    #    d_xx u_y, d_xy u_y, d_yx u_y, d_yy u_y)
    assert len(harms.shape) == 3
    assert harms.shape[-1] == 14 # u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y
    assert len(biharms.shape) == 3
    assert biharms.shape[-1] == 2 # u_x, u_y
    assert harms.shape[1] == 3935 # Dofs located at each of the 3935 vertices of the learnExt mesh.
    assert torch.linalg.norm(harms - harms[0,...]) > 0.0
    assert torch.linalg.norm(biharms - biharms[0,...]) > 0.0
    assert torch.all( torch.linalg.norm(biharms - harms[:,:,:2], dim=(1,2)) > 0.0 )

    try:
        x, y = dataset[checkpoints[0]]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    except :
        raise AssertionError
    try:
        x, y = dataset[checkpoints[1]]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    except :
        raise AssertionError
    try:
        x, y = dataset[checkpoints[-1]]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    except :
        raise AssertionError

    return
    

if __name__ == "__main__":
    test_clm_grad_dataset()
    test_clm_grad_hess_dataset()
