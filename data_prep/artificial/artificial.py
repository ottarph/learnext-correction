

class DeformationFactory:
    """
        Creates instances of deformation of the solid domain
    """
    pass

class BoundaryForceGenerator:
    """
        Generates random instances of boundary forces on
        the solid to be turned into deformed solids by
        ElasticSolver.
    """
    pass

class ElasticSolver:
    """
        Takes boundary forces on the solid domain and turns
        them into elastically deformed solid domain.

        Are deformations made by linear elasticity and 
        nonlinear elasticitiy too different for our case?
    """
    pass

class HarmonicExtender:
    """
        Takes deformations made by ``DeformationFactory`` and
        turns them into harmonic extensions over the 
        fluid domain.
    """
    pass

class BiharmonicExtender:
    """
        Takes deformations made by ``DeformationFactory`` and
        turns them into biharmonic extensions over the 
        fluid domain.
    """
    pass




