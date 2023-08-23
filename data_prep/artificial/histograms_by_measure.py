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


for i, data in enumerate(dat_scheme):

    fig, axs = plt.subplots(2, 5, sharex="col", sharey="row", layout="tight", figsize=(12,6))
    axs0 = axs[0,:]
    axs1 = axs[1,:]
    color = "black"
    alpha = 0.8


    axs0[0].axes.get_yaxis().set_ticks([])
    axs1[0].axes.get_yaxis().set_ticks([])
    axs0[0].set_ylabel("FSI dataset")
    axs1[0].set_ylabel("Artificial dataset")

    axs0[0].hist(fsi_biharm[:,0,i], bins=10, color=color, alpha=alpha)
    axs0[1].hist(fsi_biharm[:,1,i], bins=10, color=color, alpha=alpha)
    axs0[2].hist(fsi_biharm[:,2,i], bins=10, color=color, alpha=alpha)
    axs0[3].hist(fsi_biharm[:,3,i], bins=10, color=color, alpha=alpha)
    axs0[4].hist(fsi_biharm[:,4,i], bins=10, color=color, alpha=alpha)

    axs1[0].hist(art_biharm[:,0,i], bins=10, color=color, alpha=alpha)
    axs1[1].hist(art_biharm[:,1,i], bins=10, color=color, alpha=alpha)
    axs1[2].hist(art_biharm[:,2,i], bins=10, color=color, alpha=alpha)
    axs1[3].hist(art_biharm[:,3,i], bins=10, color=color, alpha=alpha)
    axs1[4].hist(art_biharm[:,4,i], bins=10, color=color, alpha=alpha)


    axs0[0].set_title("Lower flag corner")
    axs0[1].set_title("Upper flag corner")
    axs0[2].set_title("Middle flag end")
    axs0[3].set_title("Lower flag midpoint")
    axs0[4].set_title("Upper flag midpoint")

    fig.savefig(f"figures/dataset_hist_{i+1}.pdf")
    fig.suptitle(data)


plt.show()
