import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import dolfin as df



def fenics_to_scatter(u: df.Function, cmap: str = "viridis", figsize: tuple[int, int] = (12,6)) -> mpl.figure.Figure:
    """
        Plots a scatter plot of the value of u at each vertex of the 
        mesh of the function space of u. For easier visualization without
        having to download files and use paraview.

        For vector functions, takes the magnitude

    """

    mesh = u.function_space().mesh()
    x = mesh.coordinates()
    uu = u.compute_vertex_values()
    if uu.shape[0] > x.shape[0]:
        if uu.shape[0] == 2 * x.shape[0]:
            uu = np.sqrt(uu[:x.shape[0]]**2 + uu[x.shape[0]:]**2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x[:,0], x[:,1], c=uu, cmap=cmap)

    return fig

def fenics_to_scatter_moved(u: df.Function, cmap: str = "viridis", figsize: tuple[int, int] = (12,6)) -> mpl.figure.Figure:
    """
        Plots a scatter plot of the displaced vertices of the 
        mesh of the function space of u, colored by magnitude 
        of displacement. For easier visualization without
        having to download files and use paraview. Takes only 
        2D functions.

        For vector functions, takes the magnitude

    """

    mesh = u.function_space().mesh()
    x = mesh.coordinates()
    uu = u.compute_vertex_values()
    
    if not uu.shape[0] == 2 * x.shape[0]:
        raise ValueError("Only takes two-dimensional functions.")
    
    cc = np.sqrt(uu[:x.shape[0]]**2 + uu[x.shape[0]:]**2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x[:,0] + uu[:x.shape[0]], x[:,1] + uu[x.shape[0]:], c=cc, cmap=cmap)

    return fig

