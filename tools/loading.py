import numpy as np
import dolfin as df
import fem_nets


def load_mesh(mesh_file_loc: str) -> tuple[df.Mesh, df.Mesh, df.Mesh]:

    mesh = df.Mesh()
    with df.XDMFFile(mesh_file_loc + "/mesh_triangles.xdmf") as infile:
        infile.read(mesh)

    from dolfin import MeshValueCollection
    mvc = MeshValueCollection("size_t", mesh, 2)
    mvc2 = MeshValueCollection("size_t", mesh, 2)
    
    with df.XDMFFile(mesh_file_loc + "/facet_mesh.xdmf") as infile:
        infile.read(mvc, "name_to_read")
    with df.XDMFFile(mesh_file_loc + "/mesh_triangles.xdmf") as infile:
        infile.read(mvc2, "name_to_read")

    from dolfin import cpp
    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    domains = cpp.mesh.MeshFunctionSizet(mesh,mvc2)

    params = np.load(mesh_file_loc + "/params.npy", allow_pickle='TRUE').item()

    from dolfin import MeshView
    fluid_domain = MeshView.create(domains, params["fluid"])
    solid_domain = MeshView.create(domains, params["solid"])

    return mesh, fluid_domain, solid_domain


def load_mesh_submesh(mesh_file_loc: str) -> df.Mesh:

    mesh = df.Mesh()
    with df.XDMFFile(mesh_file_loc + "/mesh_triangles.xdmf") as infile:
        infile.read(mesh)


    from dolfin import MeshValueCollection
    # mvc = MeshValueCollection("size_t", mesh, 2)
    mvc2 = MeshValueCollection("size_t", mesh, 2)
    
    # with df.XDMFFile(mesh_file_loc + "/facet_mesh.xdmf") as infile:
    #     infile.read(mvc, "name_to_read")
    with df.XDMFFile(mesh_file_loc + "/mesh_triangles.xdmf") as infile:
        infile.read(mvc2, "name_to_read")

    from dolfin import cpp
    # boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    domains = cpp.mesh.MeshFunctionSizet(mesh,mvc2)


    params = np.load(mesh_file_loc + "/params.npy", allow_pickle='TRUE').item()
    # Fluid domain is tagged 7
    # Solid domain is tagged 8


    submesh = df.SubMesh(mesh, domains, params["fluid"])

    return submesh


from typing import Literal
def create_meshview_submesh_conversion_array(mesh_file_loc: str, element: Literal["CG2", "CG1"]) -> np.ndarray[int]:

    _, mesh_mv, _ = load_mesh(mesh_file_loc)
    mesh_sm = load_mesh_submesh(mesh_file_loc)

    if element == "CG2":
        V_mv = df.VectorFunctionSpace(mesh_mv, "CG", 2, 2)
        V_sm = df.VectorFunctionSpace(mesh_sm, "CG", 2, 2)
    elif element == "CG1":
        V_mv = df.VectorFunctionSpace(mesh_mv, "CG", 1, 2)
        V_sm = df.VectorFunctionSpace(mesh_sm, "CG", 1, 2)
    else:
        raise ValueError("Element not supported")
    
    ordering_f_mv = df.Function(V_mv)
    ordering_f_mv.interpolate(df.Constant((1.0, -1.0)))
    new_dofs = np.copy(ordering_f_mv.vector().get_local())
    new_dofs[::2] *= np.arange(new_dofs.shape[0] // 2)
    new_dofs[1::2] *= np.arange(new_dofs.shape[0] // 2)
    ordering_f_mv.vector().set_local(new_dofs)

    ordering_f_sm = df.Function(V_sm)
    ordering_f_sm.interpolate(ordering_f_mv)

    inds_float = ordering_f_sm.vector().get_local()[::2]
    inds = np.rint(inds_float).astype(int)

    xy_inds = np.zeros((inds.shape[0]*2,), dtype=int)
    xy_inds[::2] = 2*inds
    xy_inds[1::2] = 2*inds+1
    
    return xy_inds


def load_harmonic_data(harmonic_file_loc: str, u: df.Function, checkpoint: int = 0) -> df.Function:
    from conf import harmonic_label

    # Check u is a vector function, to avoid silent seg-fault. Looks horrible but might work for CG-spaces
    assert u.function_space().num_sub_spaces() > 0, "Is u a function in a VectorFunctionSpace?"
    assert u.function_space().dofmap().block_size() > 0, "Checks whether u.function_space() is a vector function space"

    with df.XDMFFile(harmonic_file_loc) as infile:
        infile.read_checkpoint(u, harmonic_label, checkpoint)

    return u

def load_biharmonic_data(biharmonic_file_loc: str, u: df.Function, checkpoint: int = 0) -> df.Function:
    from conf import biharmonic_label

    # Check u is a vector function, to avoid silent seg-fault. Looks horrible but might work for CG-spaces
    assert u.function_space().num_sub_spaces() > 0, "Is u a function in a VectorFunctionSpace?"
    assert u.function_space().dofmap().block_size() > 0, "Checks whether u.function_space() is a vector function space"

    with df.XDMFFile(biharmonic_file_loc) as infile:
        infile.read_checkpoint(u, biharmonic_label, checkpoint)

    return u


from typing import NewType, Any
Femnet = NewType("Femnet", Any)

def fenics_to_femnet(u: df.Function) -> Femnet:

    u_fn = fem_nets.to_torch(u.function_space())
    u_fn.double()
    u_fn.set_from_coefficients(u.vector().get_local())

    return u_fn


    # def set_from_coefficients(self, coefs):
    #     '''Set the degrees of freedom'''
    #     with torch.no_grad():
    #         self.lin.weight[0] = torch.tensor(coefs)

def femnet_to_fenics(u_fn: Femnet, V: df.FunctionSpace) -> df.Function:
    """
        Is inefficient for now, as it evaluates the network at all dof points, meaning it needs
        to build a large vandermonde matrix, instead of using coefficients which might be
        stored good enough in the network.
    """

    # u = df.Function(V)
    # u.vector().set_local(u_fn.lin.weight[0].detach().numpy())

    # return u

    if V.num_sub_spaces() > 0:
        return femnet_to_fenics_vector(u_fn, V)
    else:
        return femnet_to_fenics_scalar(u_fn, V)

def femnet_to_fenics_vector(u_fn: Femnet, V: df.FunctionSpace) -> df.Function:

    u = df.Function(V)

    new_u_dofs = np.zeros_like(u.vector()[:])
    new_u_dofs[::2] = u_fn.lins[0].weight[0].detach().numpy()
    new_u_dofs[1::2] = u_fn.lins[1].weight[0].detach().numpy()

    u.vector().set_local(new_u_dofs) # Seems to work on what it's tested on

    # dof_coordinates = V.tabulate_dof_coordinates()[::2]
    # x_torch = torch.tensor(dof_coordinates[None,...], dtype=torch.double)
    # y_torch = u_fn(x_torch)
    # y_np = y_torch[0,...].detach().numpy()

    # new_u_dofs = np.zeros(dof_coordinates.shape[0]*2, dtype=float)

    # new_u_dofs[::2] = y_np[:,0]  # Insert the x_dofs from femnet
    # new_u_dofs[1::2] = y_np[:,1] # Insert the y_dofs from femnet

    # u2 = df.Function(V)
    # u2.vector()[:] = new_u_dofs

    # return u2
    
    return u


def femnet_to_fenics_scalar(u_fn: Femnet, V: df.FunctionSpace) -> df.Function:

    u = df.Function(V)
    u.vector().set_local(u_fn.lin.weight[0].detach().numpy()) # Seems to work

    # dof_coordinates = V.tabulate_dof_coordinates()

    # x_torch = torch.tensor(dof_coordinates[None,...], dtype=torch.double)
    # y_torch = u_fn(x_torch)
    # y_np = y_torch[0,...].detach().numpy()

    # u2 = df.Function(V)
    # u2.vector()[:] = y_np
    # return u2

    return u
