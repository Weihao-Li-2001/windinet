import h5py
import matplotlib.pyplot as plt

H5_PATH = "sandbox/train.h5"

sample_name = "0000_gamma1.2200000286"
timestep = 25

variables = [
    "density",
    "momentum_x",
    "momentum_y",
    "pressure"
]

with h5py.File(H5_PATH, "r") as f:

    sample = f[sample_name]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, var in zip(axes.flat, variables):

        field = sample[var][timestep, 0]

        im = ax.imshow(field, origin="lower")

        ax.set_title(var)

        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()