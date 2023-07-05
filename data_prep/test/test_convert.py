from data_prep.convert import *


def test_cg2_vector_to_array():

    mesh = df.UnitSquareMesh(10, 10)
    V = df.VectorFunctionSpace(mesh, "CG", 2, 2)
    u = df.Function(V)
    u.interpolate(df.Constant((1.0, -1.0)))
    arr = CG2_vector_to_array(u)

    assert np.isclose(arr[:,0], 1.0).all()
    assert np.isclose(arr[:,1], -1.0).all()

    return


if __name__ == "__main__":
    test_cg2_vector_to_array()
