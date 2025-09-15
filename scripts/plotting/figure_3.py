import os

import pandas as pd
import numpy as np

from matplotlib import pylab as pl
from matplotlib import pyplot as plt

from scripts.plotting.tasks import setup_matplotlib
from src.setup.get_dir import DATA_DIR, FIG_DIR

data_dir = os.path.join(DATA_DIR, "figure_3")
fig_path = os.path.join(FIG_DIR, "figure_3.pdf")
os.makedirs(FIG_DIR, exist_ok=True)

data_files = [
    "linspace_data.csv",
    "linspace_undistorted_data.csv",
    "recursive_data.csv",
]

data_paths = [os.path.join(data_dir, file) for file in data_files]
exists = [os.path.exists(path) for path in data_paths]
if not all(exists):
    missing = [file for file, ex in zip(data_files, exists) if not ex]
    raise FileNotFoundError(f"Data files not found: {', '.join(missing)}.")

linspace_df = pd.read_csv(data_paths[0], index_col=False)
linspace_undistorted_df = pd.read_csv(data_paths[1], index_col=False)
recursive_df = pd.read_csv(data_paths[2], index_col=False)

cmap = pl.colormaps['viridis']
norm = plt.Normalize(vmin=0, vmax=1)

fig, axes = plt.subplots(7, 1, gridspec_kw={'height_ratios': [1, 1, 1, 0.5, 1, 1, 0.2]})

T = linspace_df["f"].max()

axes[0].plot([0, linspace_df["f"].max()], [0, linspace_df["f"].max()], "--", color=cmap(norm(0.2)), label="Undilated")
axes[0].plot(linspace_df["t"], linspace_df["f"], color=cmap(norm(0.6)), label="Dilation")
axins = axes[0].inset_axes([0.7025, 0.21, 0.28, 0.42],
                           xlim=(0, T),
                           ylim=(0, (linspace_df["f"]-linspace_df["t"]).max()*1.2),
                           xticks=[0, 50])
axins.tick_params(axis='both', which='major', labelsize=8)
axins.set_xticklabels(["0", "50"], horizontalalignment='right')
axins.set_ylabel(r"$t\!-\!\tau$/ns", size=8, labelpad=-0.1, y=0.4)
axins.plot(linspace_df["t"], linspace_df["f"]-linspace_df["t"], color=cmap(norm(0.6)))

# v corresponds to upper case V in the paper
axes[1].plot(linspace_undistorted_df["t"], linspace_undistorted_df["v"]/np.pi, "--", color=cmap(norm(0.2)), label=r"$\Delta^{\!\!-}V_{ij}(\tau)$")
axes[1].plot(linspace_df["t"], linspace_df["distorted_v"]/np.pi, color=cmap(norm(0.6)), label=r"$\frac{\textrm{d}f}{\textrm{d}\tau}\Delta^{\!\!-}V_{ij}\!\circ\!\!f(\tau)$")

axes[2].plot(linspace_undistorted_df["t"], linspace_undistorted_df["frequency_1_old"], color=cmap(norm(0.4)), label=r"$\tilde\omega\circ\Phi_i(\tau)$", linestyle=":")
axes[2].plot(linspace_undistorted_df["t"], linspace_undistorted_df["frequency_2_old"], "orange", label=r"$\tilde\omega\circ\Phi_j(\tau)$", linestyle=":")
axes[2].plot(linspace_undistorted_df["t"], linspace_undistorted_df["frequency_1"], label=r"$\tilde\omega\circ\Phi_i'(\tau)$", linestyle="--", alpha=0.25)
axes[2].plot(linspace_undistorted_df["t"], linspace_undistorted_df["frequency_2"], label=r"$\tilde\omega\circ\Phi_j'(\tau)$", linestyle="--", alpha=0.25)
axes[2].plot(linspace_df["t"], linspace_df["frequency_1_distorted"], color=cmap(norm(0.4)), label=r"$\tilde\omega\circ\Phi_i'\circ f(\tau)$")
axes[2].plot(linspace_df["t"], linspace_df["frequency_2_distorted"], "orange",label=r"$\tilde\omega\circ\Phi_j'\circ f(\tau)$")

axes[3].plot([0], [0], color=cmap(norm(0.4)), label=r"$\tilde\omega_i$ original", linestyle=":")
axes[3].plot([0], [0], color=cmap(norm(0.4)), label=r"$\tilde\omega_i'$ no dilation", linestyle="--", alpha=0.25)
axes[3].plot([0], [0], color=cmap(norm(0.4)), label=r"$\tilde\omega_i'$ with dilation")
axes[3].plot([0], [0], "orange", label=r"$\tilde\omega_j$ original", linestyle=":")
axes[3].plot([0], [0], "orange", label=r"$\tilde\omega_j'$ no dilation", linestyle="--", alpha=0.25)
axes[3].plot([0], [0], "orange",label=r"$\tilde\omega_j'$ with dilation")

axes[4].plot(recursive_df["t"], recursive_df["dphi_1"], label=r"$\dot\phi_i(\tau)$", color=cmap(norm(0.4)))
axes[4].plot(recursive_df["t"], recursive_df["dphi_2"], label=r"$\dot\phi_j(\tau)$", color="orange")

axes[5].plot(linspace_undistorted_df["t"], linspace_undistorted_df["I_old"], label=r"$I(\tau)$", color=cmap(norm(0.2)))
axes[5].plot(linspace_undistorted_df["t"], linspace_undistorted_df["Q_old"], label=r"$Q(\tau)$", linestyle="--", color=cmap(norm(0.2)))
axes[5].plot(linspace_df["t"], linspace_df["I_new"], label=r"$I'(\tau)$", color=cmap(norm(0.6)))
axes[5].plot(linspace_df["t"], linspace_df["Q_new"], label=r"$Q'(\tau)$",  linestyle="--", color=cmap(norm(0.6)))

axes[6].plot([0], [0], label=r"$I(\tau)$", color=cmap(norm(0.2)))
axes[6].plot([0], [0], label=r"$Q(\tau)$", linestyle="--", color=cmap(norm(0.2)))
axes[6].plot([0], [0], label=r"$I'(\tau)$", color=cmap(norm(0.6)))
axes[6].plot([0], [0], label=r"$Q'(\tau)$", linestyle="--", color=cmap(norm(0.6)))

axes[0].set_xlim([0, T])
axes[1].set_xlim([0, T])
axes[2].set_xlim([0, T])
axes[4].set_xlim([0, T])
axes[5].set_xlim([0, T])
axes[0].set_ylim([0, T])

axes[0].set_yticks([0, 25, 50])
axes[1].set_yticks([0,1,2,3])

legend = axes[0].legend(loc='upper left', handlelength=1.3, borderpad=0.3, handletextpad=0.3)
bbox = legend.get_bbox_to_anchor().transformed(axes[0].transAxes.inverted()) 
x = 0.075
bbox.x0 += x
bbox.x1 += x
legend.set_bbox_to_anchor(bbox, transform=axes[0].transAxes)
axes[1].legend(loc='lower right', handlelength=1.3, borderpad=0.3, handletextpad=0.3)
axes[3].legend(ncol=2, loc='center right', handlelength=1.3, borderpad=0.3, handletextpad=0.3)
axes[3].set_axis_off()
axes[4].legend(loc="center right", handlelength=0.7, borderpad=0.3, handletextpad=0.2, ncol=2, columnspacing=0.4)
axes[6].set_axis_off()
axes[6].legend(loc='upper center', handlelength=1.3, ncol=2, borderpad=0.3, handletextpad=0.3)

axes[0].set_ylabel("Dilation\n$t=f(\\tau)$ / ns")
axes[1].set_ylabel("Virtual $Z$ pulse\nintegrated / $\\pi$ rad")
axes[2].set_ylabel("Qubit frequencies\n$\\tilde\\omega$ / GHz")
axes[4].set_ylabel("Rotating wave\nfrequency $\\dot\\phi$ / GHz")
axes[5].set_ylabel(r"$I(\tau)$ and $Q(\tau)$"+"\n/ arbitrary units")
axes[5].set_xlabel(r"time $\tau$ / ns")

axes[0].annotate('(a)', xy=(0.01, 0.96), xycoords='axes fraction', fontsize=10, ha='left', va='top')
axes[1].annotate('(b)', xy=(0.01, 0.96), xycoords='axes fraction', fontsize=10, ha='left', va='top')
axes[2].annotate('(c)', xy=(0.01, 0.96), xycoords='axes fraction', fontsize=10, ha='left', va='top')
axes[4].annotate('(d)', xy=(0.01, 0.96), xycoords='axes fraction', fontsize=10, ha='left', va='top')
axes[5].annotate('(e)', xy=(0.01, 0.96), xycoords='axes fraction', fontsize=10, ha='left', va='top')

width = 3.40457
height = 7.05826 * 1.15
fig.set_size_inches(width, height)

fig.subplots_adjust(left=0.205, right=0.975, top=0.992, bottom=0.041, hspace=0.47)

fig.savefig(fig_path, dpi=600)