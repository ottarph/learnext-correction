import dolfin as df
import numpy as np
import pyvista as pv


class MeshQuality:

    def __init__(self, mesh: df.Mesh, quality_measure: str = "scaled_jacobian"):

        self.mesh = mesh
        self.quality_measure = quality_measure

        self.polydata = self.build_polydata(mesh)

        return
    
    def build_polydata(self, mesh: df.Mesh) -> pv.PolyData:

        points = np.column_stack((mesh.coordinates()[:,0], mesh.coordinates()[:,1], np.zeros_like(mesh.coordinates()[:,0])))
        faces = np.concatenate((3*np.ones((mesh.num_cells(), 1), dtype=np.uint32), mesh.cells()), axis=1).flatten()

        return pv.PolyData(points, faces)
    
    def convert_vector_field(self, u: df.Function) -> np.ndarray:
        assert u.function_space().ufl_element().value_shape() == (2,)
        assert u.function_space().mesh() == self.mesh

        uh_tmp = u.compute_vertex_values()

        return np.column_stack((uh_tmp[:len(uh_tmp)//2], uh_tmp[len(uh_tmp)//2:], np.zeros(len(uh_tmp)//2)))
    
    def __call__(self, u: df.Function) -> np.ndarray:
        """
        Compute the mesh quality of `self.mesh` deformed by u.

        Args:
            u (df.Function): Function to deform mesh by

        Returns:
            np.ndarray: The mesh quality of all cells in deformed mesh, ordered the same as self.mesh.cells().
        """

        self.polydata["uh"] = self.convert_vector_field(u)
        warped = self.polydata.warp_by_vector("uh")
        quality = warped.compute_cell_quality(quality_measure=self.quality_measure)
        
        return quality.cell_data["CellQuality"].base

def main():

    from pathlib import Path
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    TestFilePath = Path("../LearnExt/Output/Extension/learnext_p2_period/output.xdmf")
    test_file_label = "uh"
    num_test_checkpoints = 206
    test_file = df.XDMFFile(str(TestFilePath))

    msh = df.Mesh()
    test_file.read(msh)

    CG2 = df.VectorFunctionSpace(msh, "CG", 2)
    u_cg2 = df.Function(CG2)

    scaled_jacobian = MeshQuality(msh, "scaled_jacobian")

    min_mq = np.zeros(num_test_checkpoints)
    for k in tqdm(range(num_test_checkpoints)):
        test_file.read_checkpoint(u_cg2, test_file_label, k)
        mq = scaled_jacobian(u_cg2)
        min_mq[k] = mq.min()

    fig, ax = plt.subplots()
    ax.plot(range(num_test_checkpoints), min_mq, 'k-')
    fig.savefig("foo/figures/biharm_min_mq.pdf")

    return

if __name__ == "__main__":
    main()
