import numpy as np
import matplotlib.pyplot as plt

art_harm = np.load("data/artificial_harm_datafile.npy")
art_biharm = np.load("data/artificial_biharm_datafile.npy")
fsi_harm = np.load("data/fsi_harm_datafile.npy")
fsi_biharm = np.load("data/fsi_biharm_datafile.npy")


dat_scheme = ("u_x", "u_y", "D_x u_x", "D_y u_x", "D_x u_y", "D_y u_y")

""" Sensor locations to measure pointwise data
    place these at the corners and midpoint of the tip of the flag,
    as well as on the midpoints along the side. """

sensors = [
    np.array([0.6           , 0.19]),    # Lower corner of tip of flag
    np.array([0.6           , 0.21]),    # Upper corner of tip of flag
    np.array([0.6           , 0.20]),    # Midpoint of tip of flag
    np.array([0.424494897425, 0.19]),    # Approximate midpoint of lower side of flag
    np.array([0.424494897425, 0.21])     # Approximate midpoint of upper side of flag
]
sensor_names = [
    "Lower flag corner",
    "Upper flag corner",
    "Middle flag end",
    "Lower flag midpoint",
    "Upper flag midpoint"
]


for i, sensor_name in enumerate(sensor_names):

    fig, axs = plt.subplots(2, 6, sharex="col", sharey="row", layout="tight", figsize=(12,6))
    axs0 = axs[0,:]
    axs1 = axs[1,:]
    color = "black"
    alpha = 0.8


    axs0[0].axes.get_yaxis().set_ticks([])
    axs1[0].axes.get_yaxis().set_ticks([])
    axs0[0].set_ylabel("FSI dataset")
    axs1[0].set_ylabel("Artificial dataset")

    axs0[0].hist(fsi_biharm[:,i,0], bins=10, color=color, alpha=alpha)
    axs0[1].hist(fsi_biharm[:,i,1], bins=10, color=color, alpha=alpha)
    axs0[2].hist(fsi_biharm[:,i,2], bins=10, color=color, alpha=alpha)
    axs0[3].hist(fsi_biharm[:,i,3], bins=10, color=color, alpha=alpha)
    axs0[4].hist(fsi_biharm[:,i,4], bins=10, color=color, alpha=alpha)
    axs0[5].hist(fsi_biharm[:,i,5], bins=10, color=color, alpha=alpha)

    axs1[0].hist(art_biharm[:,i,0], bins=10, color=color, alpha=alpha)
    axs1[1].hist(art_biharm[:,i,1], bins=10, color=color, alpha=alpha)
    axs1[2].hist(art_biharm[:,i,2], bins=10, color=color, alpha=alpha)
    axs1[3].hist(art_biharm[:,i,3], bins=10, color=color, alpha=alpha)
    axs1[4].hist(art_biharm[:,i,4], bins=10, color=color, alpha=alpha)
    axs1[5].hist(art_biharm[:,i,5], bins=10, color=color, alpha=alpha)


    axs0[0].set_title("$u_x$")
    axs0[1].set_title("$u_y$")
    axs0[2].set_title(r"$\frac{\partial}{\partial x} u_x$")
    axs0[3].set_title(r"$\frac{\partial}{\partial y} u_x$")
    axs0[4].set_title(r"$\frac{\partial}{\partial x} u_y$")
    axs0[5].set_title(r"$\frac{\partial}{\partial y} u_y$")

    fig.savefig(f"figures/dataset_hist_by_sensor_{i+1}.pdf")
    fig.suptitle(sensor_name)


plt.show()
