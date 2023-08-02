
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

    indices = torch.LongTensor([0, 1])
    dim = -1
    trim_mod = TrimModule(indices, dim=dim)
    y = trim_mod(x)
    assert torch.equal(y, x[:, :, :2])

    indices = torch.LongTensor(range(2))
    dim = -1
    trim_mod = TrimModule(indices, dim=dim)
    y = trim_mod(x)
    assert torch.equal(y, x[:, :, :2])

    indices = torch.LongTensor(range(10))
    dim = -2
    trim_mod = TrimModule(indices, dim=dim)
    y = trim_mod(x)
    assert torch.equal(y, x[:, :10, :])

    inds1 = torch.LongTensor(range(2))
    inds2 = torch.LongTensor(range(10))
    trim_mod1 = TrimModule(inds1, dim=-1)
    trim_mod2 = TrimModule(inds2, dim=-2)
    trim_mod_stack = nn.Sequential(trim_mod1, trim_mod2)
    y = trim_mod_stack(x)
    assert torch.equal(y, x[:, :10, :2])

    inds1 = torch.LongTensor(range(1, 3))
    inds2 = torch.LongTensor(range(5, 15))
    trim_mod1 = TrimModule(inds1, dim=-1)
    trim_mod2 = TrimModule(inds2, dim=-2)
    trim_mod_stack = nn.Sequential(trim_mod1, trim_mod2)
    y = trim_mod_stack(x)
    assert torch.equal(y, x[:, 5:15, 1:3])

    inds1 = torch.LongTensor(range(1, 3))
    inds2 = torch.LongTensor(range(2, 13, 2))
    trim_mod1 = TrimModule(inds1, dim=-1)
    trim_mod2 = TrimModule(inds2, dim=-2)
    trim_mod_stack = nn.Sequential(trim_mod1, trim_mod2)
    y = trim_mod_stack(x)
    assert torch.equal(y, x[:, 2:13:2, 1:3])

    inds = torch.LongTensor(range(1, 5, 2))
    trim_mod = TrimModule(inds, dim=-1)
    y = trim_mod(x)
    assert torch.equal(y, x[:, :, 1:5:2])

    if torch.cuda.is_available():
        device = "cuda"
        indices = torch.LongTensor([0, 1])
        dim = -1
        trim_mod = TrimModule(indices, dim=dim)
        trim_mod.to(device)
        x_dev = x.to(device)
        y_dev = trim_mod(x_dev)
        assert torch.equal(y_dev, x_dev[:, :, :2])

    return

def test_prepend_cat():

    b = 1
    n = 10
    m1 = 2
    m2 = 4
    x = torch.rand((b, n, m1))
    y = torch.rand((b, n, m2))
    z = torch.cat((x, y), dim=-1)
    w = torch.zeros_like(z)
    for i in range(w.shape[-2]):
        for j in range(x.shape[-1]):
            w[...,i,j] = x[...,i,j]
        for j in range(y.shape[-1]):
            w[...,i,x.shape[-1]+j] = y[...,i,j]
    assert torch.equal(z, w)

    b2 = 2
    x = torch.rand((b, b2, n, m1))
    y = torch.rand((b, b2, n, m2))
    z = torch.cat((x, y), dim=-1)
    w = torch.zeros_like(z)
    for i in range(w.shape[-2]):
        for j in range(x.shape[-1]):
            w[...,i,j] = x[...,i,j]
        for j in range(y.shape[-1]):
            w[...,i,x.shape[-1]+j] = y[...,i,j]
    assert torch.equal(z, w)

    return

def test_prepend_expand():

    prep = torch.rand(10, 2)
    x = torch.rand(5, 6, 3, 10, 4)
    new_prep = prep.reshape([1]*(len(x.shape)-len(prep.shape)) + [*prep.shape])
    assert new_prep.shape == (1, 1, 1, 10, 2)

    new_prep = prep.expand([*x.shape[:-1]]+[prep.shape[-1]])
    assert new_prep.shape == (5, 6, 3, 10, 2)

    y = torch.cat((new_prep, x), dim=-1)
    assert torch.linalg.norm(y[...,:,:2] - prep).item() == 0.0
    assert torch.linalg.norm(y[...,:,2:] - x).item() == 0.0

    return

def test_prepend_module():


    batch = 5
    N = 10
    a = torch.rand((batch, N))
    b = torch.rand((batch, N))
    prepend_tensor = torch.zeros((batch, N, 2))
    prepend_tensor[...,0] = a
    prepend_tensor[...,1] = b

    x = torch.rand((batch, N))
    y = torch.rand((batch, N))
    z = torch.rand((batch, N))
    w = torch.rand((batch, N))
    inp_vec = torch.zeros((batch, N, 4))
    inp_vec[...,0] = x
    inp_vec[...,1] = y
    inp_vec[...,2] = z
    inp_vec[...,3] = w

    prep_mod = PrependModule(prepend_tensor=prepend_tensor)
    out_vec = prep_mod(inp_vec)

    assert torch.equal(out_vec[...,:2], prepend_tensor)
    assert torch.equal(out_vec[...,2:], inp_vec)

    batch = 5
    N = 10
    a = torch.rand((N,))
    b = torch.rand((N,))
    prepend_tensor = torch.zeros((N, 2))
    prepend_tensor[:,0] = a
    prepend_tensor[:,1] = b

    x = torch.rand((batch, N))
    y = torch.rand((batch, N))
    z = torch.rand((batch, N))
    w = torch.rand((batch, N))
    inp_vec = torch.zeros((batch, N, 4))
    inp_vec[...,0] = x
    inp_vec[...,1] = y
    inp_vec[...,2] = z
    inp_vec[...,3] = w

    prep_mod = PrependModule(prepend_tensor=prepend_tensor)
    out_vec = prep_mod(inp_vec)

    assert torch.linalg.norm(out_vec[...,:,:2] - prepend_tensor).item() == 0.0
    assert torch.linalg.norm(out_vec[...,:,2:] - inp_vec).item() == 0.0

    prep_tens = torch.rand(10, 2)
    prep_mod = PrependModule(prep_tens)
    inp_vec = torch.rand(5, 3, 4, 5, 10, 4)
    out_vec = prep_mod(inp_vec)

    assert torch.linalg.norm(out_vec[...,:,:2] - prep_tens).item() == 0.0
    assert torch.linalg.norm(out_vec[...,:,2:] - inp_vec).item() == 0.0

    return


if __name__ == "__main__":
    test_mlp()
    test_tensor_module()
    test_trim_module()
    test_prepend_cat()
    test_prepend_expand()
    test_prepend_module()
