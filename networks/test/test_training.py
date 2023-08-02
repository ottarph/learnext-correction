from networks.training import *


def test_context():
    from networks.general import MLP

    widths = [2, 4, 4, 2]
    activation = nn.ReLU()
    mlp = MLP(widths, activation=activation)

    cf = nn.MSELoss()
    opt = torch.optim.SGD(mlp.parameters(), lr=1e-2)

    context = Context(mlp, cf, opt)

    """ Inserting some dummy-training info. """
    context.train_hist.append(4e-2); context.lr_hist.append(1e-3)
    context.val_hist.append(4.5e-2); context.epoch += 1
    context.train_hist.append(4e-3); context.lr_hist.append(1e-3)
    context.val_hist.append(4.5e-3); context.epoch += 1
    context.train_hist.append(4e-4); context.lr_hist.append(1e-3)
    context.val_hist.append(4.5e-4); context.epoch += 1

    context.train_hist.append(2e-4); context.lr_hist.append(1e-3)
    context.val_hist.append(4.5e-4); context.epoch += 1
    context.train_hist.append(2e-5); context.lr_hist.append(1e-3)
    context.val_hist.append(4.5e-5); context.epoch += 1
    context.train_hist.append(2e-6); context.lr_hist.append(1e-3)
    context.val_hist.append(4.5e-6); context.epoch += 1

    """ Change a weight. """
    mlp.layers[0].weight.data += 1.0

    folder_name = "networks/test/test_data"
    context.save_results(folder_name)
    context.save_model(folder_name)

    new_mlp = MLP(widths, activation=activation)
    new_context = Context(new_mlp, cf, opt)

    new_context.load_results(folder_name)
    new_context.load_model(folder_name)

    from os import remove
    remove(folder_name+"/train.txt")
    remove(folder_name+"/val.txt")
    remove(folder_name+"/lr.txt")
    remove(folder_name+"/model.txt")
    remove(folder_name+"/state_dict.pt")

    assert context.epoch == new_context.epoch

    def check_list(list1: list[float], list2: list[float]) -> bool:
        for a, b in zip(list1, list2):
            if not np.isclose(a, b):
                return False
        return True

    assert check_list(context.train_hist, new_context.train_hist)
    assert check_list(context.val_hist, new_context.val_hist)
    assert check_list(context.lr_hist, new_context.lr_hist)

    return

if __name__ == "__main__":
    test_context()
