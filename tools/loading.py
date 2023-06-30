import numpy as np
import dolfin as df


def load_mesh(mesh_file_loc: str):

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


def load_harmonic_data(data_file_loc: str, u: df.Function, checkpoint: int = 0):

    with df.XDMFFile(data_file_loc + "/input_.xdmf") as infile:
        infile.read_checkpoint(u, "input", checkpoint)

    return u

def load_biharmonic_data(data_file_loc: str, u: df.Function, checkpoint: int = 0):

    with df.XDMFFile(data_file_loc + "/output_.xdmf") as infile:
        infile.read_checkpoint(u, "output", checkpoint)

    return u



