
from networks.masknet import *
from networks.general import MLP

def test_femnet_masknet():

    torch.set_default_dtype(torch.float64)

    import dolfin as df
    from tools.loading import load_mesh, load_biharmonic_data, \
                              fenics_to_femnet
    from conf import mesh_file_loc, biharmonic_file_loc, vandermonde_loc

    _, fluid_mesh, _ = load_mesh(mesh_file_loc)

    V = df.VectorFunctionSpace(fluid_mesh, "CG", 2, 2)
    V_scal = df.FunctionSpace(fluid_mesh, "CG", 2)
    u_bih = df.Function(V)
    load_biharmonic_data(biharmonic_file_loc, u_bih, 0)

    u_g_df = laplace_extension(u_bih)
    m_df = poisson_mask(V_scal)

    u_g_fn = fenics_to_femnet(u_g_df)
    m_fn = fenics_to_femnet(m_df)

    eval_coords = torch.tensor(V_scal.tabulate_dof_coordinates())

    net = TensorModule(torch.zeros(eval_coords.shape))

    masknet = FemNetMasknet(net, u_g_fn, m_fn)
    masknet.load_vandermonde(vandermonde_loc)

    pred = masknet(eval_coords[None,...])

    base_array_raw = np.copy(u_g_df.vector()[:])
    base_array = np.column_stack((base_array_raw[::2], base_array_raw[1::2]))
    assert np.isclose(base_array, pred.detach().numpy()[0,...]).all()

    net = MLP([2, 4, 4, 2], activation=nn.ReLU())

    masknet = FemNetMasknet(net, u_g_fn, m_fn)
    masknet.load_vandermonde(vandermonde_loc)

    summ = torch.zeros((1,))
    summ += masknet(eval_coords[None,...]).sum()

    return

    
def test_tensor_masknet():
    torch.set_default_dtype(torch.float64)

    import dolfin as df
    from tools.loading import load_mesh, load_biharmonic_data
    from conf import mesh_file_loc, biharmonic_file_loc

    _, fluid_mesh, _ = load_mesh(mesh_file_loc)

    V = df.VectorFunctionSpace(fluid_mesh, "CG", 2, 2)
    V_scal = df.FunctionSpace(fluid_mesh, "CG", 2)
    u_bih = df.Function(V)
    load_biharmonic_data(biharmonic_file_loc, u_bih, 0)

    eval_coords_np = V_scal.tabulate_dof_coordinates()

    eval_coords = torch.tensor(eval_coords_np)

    u_g_df = laplace_extension(u_bih)
    m_df = poisson_mask(V_scal)

    base_array_raw = np.copy(u_g_df.vector()[:])
    base_array = np.column_stack((base_array_raw[::2], base_array_raw[1::2]))
    base_tensor = torch.tensor(base_array)

    mask_array_raw = np.copy(m_df.vector()[:])
    mask_array = mask_array_raw[...,None]
    mask_tensor = torch.tensor(mask_array)

    net = MLP([2, 4, 2], activation=nn.ReLU())
    masknet = TensorMaskNet(net, base_tensor, mask_tensor)

    summ = torch.zeros((1,))
    summ += masknet(eval_coords).sum()
    summ += masknet(eval_coords[None,None,...]).sum()
    # print(masknet(eval_coords))
    # print(masknet(eval_coords[None,None,...]))
    # print(summ)

    return



if __name__ == "__main__":
    test_femnet_masknet()
    test_tensor_masknet()

