from networks.deeponet import *
from networks.general import *
from data_prep.dataset import learnextDataset

"""
    TODO: Need to find boundary dofs to get sensors.
"""

run_params = {
    "input_domain_dim": 2,
    "input_range_dim": 2,
    "output_domain_dim": 2,
    "output_range_dim": 2,
    "function_batch": 8,
    "num_sens": 7,
    "num_eval": 6,
    "width": 4,
    "net_max_width": 10
}

def main():

    prefix = "data_prep/data_store/learnextCG2"
    checkpoints = range(0, 2400+1)

    dataset = learnextDataset(prefix, checkpoints)

    from torch.utils.data import DataLoader

    # torch.manual_seed(0)
    batch_size = 8
    shuffle = False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    harms, biharms = next(iter(dataloader))

    print(harms.shape)
    print(biharms.shape)

    input_domain_dim = run_params["input_domain_dim"]
    input_range_dim = run_params["input_range_dim"]
    output_domain_dim = run_params["output_domain_dim"]
    output_range_dim = run_params["output_range_dim"]
    function_batch = run_params["function_batch"]
    num_sens = run_params["num_sens"]
    num_eval = run_params["num_eval"]
    width = run_params["width"]
    net_max_width = run_params["net_max_width"]

    return


if __name__ == "__main__":
    main()
