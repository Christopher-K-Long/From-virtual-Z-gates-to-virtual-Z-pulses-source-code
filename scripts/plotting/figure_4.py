import os

import numpy as np
import pandas as pd

from matplotlib import pylab as pl
from matplotlib import pyplot as plt

from scripts.plotting.tasks import setup_matplotlib
from src.setup.get_dir import DATA_DIR, FIG_DIR

data_dir = os.path.join(DATA_DIR, "figure_4")
fig_path = os.path.join(FIG_DIR, "figure_4.pdf")
os.makedirs(FIG_DIR, exist_ok=True)

data_files = [
    "infidelity_gaussian_gradient.txt",
    "infidelity_gaussian.csv",
    "infidelity_tanh2_gradient.txt",
    "infidelity_tanh2.csv"
]

data_paths = [os.path.join(data_dir, file) for file in data_files]
exists = [os.path.exists(path) for path in data_paths]
if not all(exists):
    missing = [file for file, ex in zip(data_files, exists) if not ex]
    raise FileNotFoundError(f"Data files not found: {', '.join(missing)}.")

with open(data_paths[0], "r") as file:
    gaussian_gradient = float(file.read().strip())

with open(data_paths[2], "r") as file:
    tanh2_gradient = float(file.read().strip())

df_gaussian = pd.read_csv(data_paths[1], index_col=False)
df_tanh = pd.read_csv(data_paths[3], index_col=False)

cmap = pl.colormaps['viridis']
norm = plt.Normalize(vmin=0, vmax=1)

fig, axis = plt.subplots(1,1)

axis.axvline(30, linestyle="--", color="k", label="Difference in qubit frequencies")
axis.loglog(df_gaussian["max_v"], df_gaussian["infidelity"], marker="o", label=f"Gaussian (gradient=${np.round(gaussian_gradient, 2)}$)", color=cmap(norm(0.4)))
axis.loglog(df_tanh["max_v"], df_tanh["infidelity"], marker="o", label=f"Tanh (gradient=${np.round(tanh2_gradient, 2)}$)", color="orange")
axis.set_xlabel("Virtual $Z$ pulse amplitude / MHz")
axis.set_ylabel("Infidelity")
axis.legend()
width = 7.05826
height = 7.05826 * 0.4
fig.set_size_inches(width, height)
fig.tight_layout()
fig.savefig(fig_path)