import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import pylab as pl

from scripts.plotting.tasks import setup_matplotlib
from src.setup.get_dir import DATA_DIR, FIG_DIR

data_dir = os.path.join(DATA_DIR, "figure_2")
fig_path = os.path.join(FIG_DIR, "figure_2.pdf")
os.makedirs(FIG_DIR, exist_ok=True)

data_path = os.path.join(data_dir, "data.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}.")
df = pd.read_csv(data_path, index_col=False)

fig, axes_all = plt.subplots(4, 3)

for grouped, axes in zip(df.groupby("name"), axes_all.T):
    name, group = grouped

    if name == "gaussian":
        x = 0.015
        axins = axes[1].inset_axes([x, 0.515, 0.485 - x, 0.465],
                                   xlim=(0.375, 0.675),
                                   ylim=(0.375, 0.675),
                                   xticklabels=[],
                                   yticklabels=[])
        axins.set_xticks([])
        axins.set_yticks([])
    amplitudes = np.linspace(0, 1, 5)
    cmap = pl.colormaps['viridis']
    cmap.set_under("black")
    norm = plt.Normalize(vmin=amplitudes[0], vmax=amplitudes[-1])

    

    for amp, sub_group in group.groupby("amplitude"):
        # v corresponds to upper case V in the paper
        v_diff = np.diff(sub_group["v"])
        f = sub_group["f"].to_numpy()

        axes[0].plot(0.5 * (f[:-1] + f[1:]), v_diff / (f[1] - f[0]), color=cmap(norm(amp)))
        axes[1].plot(sub_group["t"], f, color=cmap(norm(amp)))
        if name == "gaussian":
            axins.plot(sub_group["t"], f, color=cmap(norm(amp)))
        axes[2].plot(sub_group["t"], sub_group["J"], color=cmap(norm(amp)))


        axes[3].plot(sub_group["t"], sub_group["I"], color=cmap(norm(amp)))
        axes[3].plot(sub_group["t"], sub_group["Q"], color=cmap(norm(amp)), linestyle="--")


    if name == "gaussian":
        axes[1].indicate_inset_zoom(axins, edgecolor="black")
        
        axes[0].set_ylabel("Virtual $Z$ pulse\namplitude")
        axes[1].set_ylabel(r"$t=f(\tau)$")
        axes[2].set_ylabel(r"$J_{ij}'(\tau)$")
        axes[3].set_ylabel(r"$I'(\tau)$ and $Q'(\tau)$")

    axes[0].set_xlabel(r"$t$")
    axes[1].set_xlabel(r"$\tau$")
    axes[2].set_xlabel(r"$\tau$")
    axes[3].set_xlabel(r"$\tau$")

axes_all[0, 0].set_title("Gaussian")
axes_all[0, 1].set_title("Tanh")
axes_all[0, 2].set_title("Tanh larger amplitude")


width = 7.05826
height = width
fig.set_size_inches(width, height)

plt.tight_layout()

fig.savefig(fig_path, dpi=600)