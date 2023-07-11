
from networks.general import *
torch.manual_seed(0)

def test_mlp():

    widths = [2, 4, 4, 2]
    activation = nn.ReLU()
    mlp = MLP(widths, activation=activation)

    summ = torch.zeros((1,))
    x = torch.rand((4, 2))
    summ += mlp(x).sum()
    x = torch.rand((1, 4, 2))
    summ += mlp(x).sum()
    x = torch.rand((3, 4, 2))
    summ += mlp(x).sum()

    return

def test_tensor_module():

    x = torch.tensor([[1.0, 2.0],[3.0, 4.0], [5.0, 6.0]])
    tm = TensorModule(x)

    assert torch.equal(tm(x), x)

    summ = torch.zeros((1,))
    y = tm(torch.tensor([1.0, 2.0]))
    summ += y.sum()
    y = tm(torch.tensor([[1.0, 2.0]]))
    summ += y.sum()

    return

def test_trim_module():

    # Testing batch of three twenty-vertex (u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y), as in clement grad net.
    x = torch.rand((3, 20, 6))

    forward_indices = [range(20), range(2)]
    trim_mod = TrimModule(forward_indices=forward_indices)
    y = trim_mod(x)
    assert torch.equal(y, x[:, :, :2])

    forward_indices = [range(10), range(2)]
    trim_mod = TrimModule(forward_indices=forward_indices)
    y = trim_mod(x)
    assert torch.equal(y, x[:, :10, :2])

    forward_indices = [range(5, 15), range(1, 3)]
    trim_mod = TrimModule(forward_indices=forward_indices)
    y = trim_mod(x)
    assert torch.equal(y, x[:, 5:15, 1:3])

    forward_indices = [range(2, 13, 2), range(1, 3)]
    trim_mod = TrimModule(forward_indices=forward_indices)
    y = trim_mod(x)
    assert torch.equal(y, x[:, 2:13:2, 1:3])

    forward_indices = [range(1, 5, 2)]
    trim_mod = TrimModule(forward_indices=forward_indices)
    y = trim_mod(x)
    assert torch.equal(y, x[:, :, 1:5:2])


    return

def test_context():

    widths = [2, 4, 4, 2]
    activation = nn.ReLU()
    mlp = MLP(widths, activation=activation)

    cf = nn.MSELoss()
    opt = torch.optim.SGD(mlp.parameters(), lr=1e-2)

    context = Context(mlp, cf, opt)

    """ Inserting some dummy-training info. """
    context.train_hist.append(4e-2); context.epoch += 1
    context.train_hist.append(4e-3); context.epoch += 1
    context.train_hist.append(4e-4); context.epoch += 1
    context.test_hist[context.epoch] = 3e-3

    context.train_hist.append(2e-4); context.epoch += 1
    context.train_hist.append(2e-5); context.epoch += 1
    context.train_hist.append(2e-6); context.epoch += 1
    context.test_hist[context.epoch] = 3e-4

    """ Change a weight. """
    mlp.layers[0].weight.data += 1.0

    fname = "networks/test/test_data/test_context"
    context.save(fname=fname)

    new_mlp = MLP(widths, activation=activation)

    new_context = Context(new_mlp, cf, opt)
    new_context.load(fname=fname)

    from os import remove
    remove(fname+".train.txt")
    remove(fname+".test.txt")
    remove(fname+".pt")

    assert context.epoch == new_context.epoch

    def check_train_hist(ctx1: Context, ctx2: Context):
        for a, b in zip(ctx1.train_hist, ctx2.train_hist):
            if not np.isclose(a, b):
                return False
        return True
    def check_test_hist(ctx1: Context, ctx2: Context):
        tested_epochs1 = list(ctx1.test_hist.keys())
        tested_epochs2 = list(ctx2.test_hist.keys())
        tested_epochs1.sort()
        tested_epochs2.sort()
        if not len(tested_epochs1) == len(tested_epochs2):
            return False
        for e1, e2 in zip(tested_epochs1, tested_epochs2):
            if e1 != e2:
                return False
        for e in tested_epochs1:
            if not np.isclose(ctx1.test_hist[e], ctx2.test_hist[e]):
                return False
        return True
    
    assert check_train_hist(context, new_context)
    assert check_test_hist(context, new_context)

    return


if __name__ == "__main__":
    test_mlp()
    test_tensor_module()
    test_trim_module()
    test_context()
