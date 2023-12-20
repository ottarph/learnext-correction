import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from pathlib import Path

def plot_min_mesh_quality_over_time(datafile: os.PathLike):

    # (#runs, #steps, #cells)
    mesh_quality_over_runs = np.load(datafile)

    mesh_quality_mins = mesh_quality_over_runs.min(axis=2)

    fig, ax = plt.subplots()

    for run in range(mesh_quality_mins.shape[0]):
        ax.plot(range(mesh_quality_mins.shape[1]), mesh_quality_mins[run], label=f"Run #{run+1}")

    ax.legend()

    return fig, ax


datafile = Path("parameter_study/data/random_init.npy")

fig, ax = plot_min_mesh_quality_over_time(datafile)

fig.savefig("analysis.pdf")

