import torch
import torch.nn as nn
import numpy as np


class DofPermutationTransform(nn.Module):

    def __init__(self, permutation_tensor: torch.Tensor | str, dim: int = -2):
        """
            Permutes the ordering of the DOF-locations, which are different depending
            on whether the mesh was loaded with MeshView or with SubMesh. This functionality
            is only meant for CG function spaces

            DOFs are stored as a vector [u_1_x, u_1_y, u_2_x, u_2_y, ..., u_N_x, u_N_y],
            where N is the number of distinct DOF-locations. 

            For CG1 function spaces, the DOF-locations are precisely the vertices of the mesh,
            and each location holds a DOF for u_x and u_y.
            
            For CG2 function spaces, the DOF-locations are the vertices and facet midpoints, and 
            each location holds a DOF for u_x and u_y.
        """
        super().__init__()

        self.permutation_tensor = permutation_tensor
        self.permutation_tensor.requires_grad_(False)

        self.dim = dim

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return torch.index_select(x, self.dim, self.permutation_tensor)
    
