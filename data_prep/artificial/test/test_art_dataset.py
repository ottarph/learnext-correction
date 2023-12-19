from data_prep.artificial.dataset import *

def test_artificial_dataset():
    prefix = "data_prep/artificial/data_store/grad/art"
    checkpoints = range(0, 606)

    dataset = ArtificialLearnextDataset(prefix, checkpoints)

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
    assert harms.shape[1] == 3924 # Dofs located at each of the 3924 vertices of the new learnExt mesh by Miro.
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
    test_artificial_dataset()
