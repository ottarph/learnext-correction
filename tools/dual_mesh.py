from xii.meshing.make_mesh_cpp import make_mesh

from itertools import combinations, count
import numpy as np


def inner_point_refine(mesh, new_pts, strict, nrefs=1):
    r'''
    Mesh refine by adding point inside each celll

       x                      x   
      / \     becomes        /|\  
     /   \                  / x \
    /     \                / / \ \  
    x------x              x-------x

    Strict is the rel tol for checing whether new_pts[i] is inside cell[i]
    '''
    if nrefs > 1:
        root_id = mesh.id()
        tdim = mesh.topology().dim()
        
        mesh0 = inner_point_refine(mesh, new_pts, strict, 1)
        nrefs -= 1
        # Root will stay the same for those are the vertices that
        # were originally in the mesh and only those can be traced
        ref_vertices, root_vertices = zip(*mesh0.parent_entity_map[mesh.id()][0].items())
        
        while nrefs > 0:
            nrefs -= 1
            
            mesh1 = inner_point_refine(mesh0, new_pts, strict, 1)
            # Upda mesh1 mapping
            mapping0, = mesh0.parent_entity_map.values()
            mapping1, = mesh1.parent_entity_map.values()
            
            new_mapping = {}
            # New cells fall under some parent
            e_mapping0, e_mapping1 = mapping0[tdim], mapping1[tdim]
            new_mapping[tdim] = {k: e_mapping0[v] for k, v in e_mapping1.items()}
            # But that's not the case with vertices, we only look for root
            # ones in the new
            e_mapping1 = mapping1[0]
            ref_vertices = [ref_v for ref_v, coarse_v in e_mapping1.items()
                            if coarse_v in ref_vertices]
            assert len(ref_vertices) == len(root_vertices)
            new_mapping[0] = dict(zip(ref_vertices, root_vertices))
                
            mesh1.parent_entity_map = {root_id: new_mapping}
            
            mesh0 = mesh1

        return mesh0

    # One refinement
    x = mesh.coordinates()
    cells = mesh.cells()

    ncells, nvtx_cell = cells.shape
    assert any((nvtx_cell == 2,
                nvtx_cell == 3,
                nvtx_cell == 4 and mesh.topology().dim() == 3))

    # Center points will be new coordinates
    xnew = new_pts(mesh)
    # We have to check for duplicates
    #if not unique_guarantee:
    #    pass
    if strict > 0:
        tol = mesh.hmin()*strict
        # The collision
        assert point_is_inside(xnew, x[cells], tol)
        
    # Each cell gives rise to ...
    child2parent = np.empty(ncells*nvtx_cell, dtype='uintp')
    fine_cells = np.empty((ncells*nvtx_cell, nvtx_cell), dtype='uintp')
    # How we build new cells
    basis = list(map(list, combinations(list(range(nvtx_cell)), nvtx_cell-1)))

    fine_coords = np.row_stack([x, xnew])
    
    fc, center = 0, len(x)
    for pc, cell in enumerate(cells):
        for base in basis:
            new_cell = np.r_[cell[base], center]
            # Every new cell must be non-empty
            assert simplex_area(fine_coords[new_cell]) > 1E-15

            fine_cells[fc, :] = new_cell
            child2parent[fc] = pc
            fc += 1
        center += 1

    tdim = mesh.topology().dim()
                                              
    fine_mesh = make_mesh(fine_coords, fine_cells, tdim, mesh.geometry().dim())

    fine_mesh.parent_entity_map = {mesh.id(): {0: dict(enumerate(range(mesh.num_vertices()))),
                                               tdim: dict(enumerate(child2parent))}}
       
    return fine_mesh
    

def centroid_refine(mesh, strict=1E-10, nrefs=1):
    '''Using centroid'''
    # How to
    # new_pts = lambda mesh: np.mean(mesh.coordinates()[mesh.cells()], axis=1)
    def new_pts(mesh):
        ncells = mesh.num_cells()
        new_pts = np.mean(mesh.coordinates()[mesh.cells()], axis=1)

        return new_pts
       
    return inner_point_refine(mesh, new_pts, strict, nrefs=nrefs)



def point_is_inside(x, cell, tol):
    '''Is x inside a simplex cell'''
    # Many
    if x.ndim > 1:
        return all(point_is_inside(*pair, tol=tol) for pair in zip(x, cell))
    # We say the point is inside if the area of small cells made using
    # it adds up to the area of the cell
    diff = abs(simplex_area(cell) - sum(map(simplex_area, simplices(x, cell))))
    return diff < tol


def simplices(x, cell):
    '''Insert x to cell'''
    # Subdivide interval AB -> AX, BX
    nvtx = len(cell)
    for other in map(list, combinations(list(range(nvtx)), nvtx-1)):
        yield np.row_stack([x, cell[other]])

        
def simplex_area(cell):
    '''The name says it all'''
    if len(cell) == 2:
        return np.linalg.norm(np.diff(cell, axis=0))

    if len(cell) == 3:
        A, B, C = cell
        if len(A) == 2:
            return np.abs(np.linalg.det(np.column_stack([cell, np.ones(3)]))/2)
        else:
            return np.linalg.norm(np.cross(B-A, C-A))/2.

    if len(cell) == 4:
        return np.abs(np.linalg.det(np.column_stack([cell, np.ones(4)]))/6.)
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    import dolfin as df
    
    mesh = df.UnitSquareMesh(1, 1)
    dual_mesh = centroid_refine(mesh)

    df.File('mesh.pvd') << mesh
    df.File('dual_mesh.pvd') << dual_mesh
