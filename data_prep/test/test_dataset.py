from data_prep.dataset import *

def test_dataset():
    prefix = "data_prep/data_store/learnextCG2"
    checkpoints = range(0, 2400+1)

    dataset = learnextDataset(prefix, checkpoints)

    from torch.utils.data import DataLoader

    # torch.manual_seed(0)
    batch_size = 8
    shuffle = False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    harms, biharms = next(iter(dataloader))

    assert len(harms.shape) == 3
    assert harms.shape[-1] == 2
    assert torch.linalg.norm(harms - harms[0,...]) > 0.0
    assert torch.linalg.norm(biharms - biharms[0,...]) > 0.0
    assert torch.all( torch.linalg.norm(biharms - harms, dim=(1,2)) > 0.0 )
    

if __name__ == "__main__":
    test_dataset()
