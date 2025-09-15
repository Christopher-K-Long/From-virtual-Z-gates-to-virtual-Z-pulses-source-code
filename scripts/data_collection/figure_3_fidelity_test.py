import os

import numpy as np
import pandas as pd
import py_ste

from scipy.linalg import expm

from src.setup.get_dir import DATA_DIR
from scripts.data_collection.tasks.pauli_strings import IZ, ZI, IX, XX, YY, ZZ

data_dir = os.path.join(DATA_DIR, "figure_3")
figure_3_data_dir = os.path.join(DATA_DIR, "figure_3")

if (not os.path.exists(os.path.join(figure_3_data_dir, "linspace_data.csv"))
 or not os.path.exists(os.path.join(figure_3_data_dir, "linspace_undistorted_data.csv"))
 or not os.path.exists(os.path.join(figure_3_data_dir, "recursive_data.csv"))):
    print("Data for figure 3 not found. Collecting data for figure 3:")
    from scripts.data_collection import figure_3
    print("Data for figure 3 collected.")

H0 = np.zeros((4, 4), dtype=complex)

evolver: py_ste.evolvers.DenseUnitaryEvolver \
    = py_ste.get_unitary_evolver(H0, [IZ, ZI, IX, XX+YY+ZZ])

linspace_df = pd.read_csv(os.path.join(data_dir, "linspace_data.csv"), index_col=False)
linspace_undistorted_df = pd.read_csv(os.path.join(data_dir, "linspace_undistorted_data.csv"), index_col=False)
recursive_df = pd.read_csv(os.path.join(data_dir, "recursive_data.csv"), index_col=False)

resonator_frequency = 2 * np.pi * 8
J_scale = 2 * np.pi * 0.01
IQ_scale = 2 * np.pi * 0.01

t = linspace_undistorted_df["t"]
frequency_1 = linspace_undistorted_df["frequency_1_old"] * 2 * np.pi
frequency_2 = linspace_undistorted_df["frequency_2_old"] * 2 * np.pi
# v corresponds to upper case V in the paper and dv to lower case v in the paper
dv = linspace_undistorted_df["dv"] * 2 * np.pi
g = linspace_undistorted_df["I_old"] * np.cos(frequency_1[0] * t) \
   +linspace_undistorted_df["Q_old"] * np.sin(frequency_1[0] * t)
J = 1 / (frequency_1 - resonator_frequency) \
   +1 / (frequency_2 - resonator_frequency)
physical_ctrl_amp = np.stack([0.5 * frequency_1 + 0.25 * dv,
                              0.5 * frequency_2 - 0.25 * dv,
                              IQ_scale * g,
                              J_scale * J],
                             axis=-1)
physical_ctrl_amp = physical_ctrl_amp.astype(complex)
physical_ctrl_amp = physical_ctrl_amp[:-1]

U_physical = evolver.get_evolution(
                physical_ctrl_amp,
                linspace_undistorted_df["t"][1]-linspace_undistorted_df["t"][0])

no_z_ctrl_amp = np.stack([0.5 * frequency_1, 0.5 * frequency_2, IQ_scale * g, J_scale * J], axis=-1)
no_z_ctrl_amp = no_z_ctrl_amp.astype(complex)
no_z_ctrl_amp = no_z_ctrl_amp[:-1]

U_no_z = evolver.get_evolution(
                no_z_ctrl_amp,
                linspace_undistorted_df["t"][1]-linspace_undistorted_df["t"][0])
U_no_z = expm(-1j * 0.25 * linspace_df["v"].to_numpy()[-1] * (IZ - ZI)) @ U_no_z


t = linspace_df["t"]
frequency_1 = linspace_df["frequency_1_distorted"] * 2 * np.pi
frequency_2 = linspace_df["frequency_2_distorted"] * 2 * np.pi
g = linspace_df["I_new"] * np.cos(frequency_1[0] * t) + linspace_df["Q_new"] * np.sin(frequency_1[0] * t)
J = 1 / (frequency_1 - resonator_frequency) + 1 / (frequency_2 - resonator_frequency)
virtual_ctrl_amp = np.stack([0.5 * frequency_1, 0.5 * frequency_2, IQ_scale * g, J_scale * J], axis=-1)
virtual_ctrl_amp = virtual_ctrl_amp.astype(complex)
virtual_ctrl_amp = virtual_ctrl_amp[:-1]

U_virtual = evolver.get_evolution(
                virtual_ctrl_amp,
                linspace_df["t"][1]-linspace_df["t"][0])

t = recursive_df["t"].to_numpy()
phi_1 = recursive_df["phi_1"].to_numpy()
phi_2 = recursive_df["phi_2"].to_numpy()
phase_1 = np.interp(linspace_df["t"].to_numpy()[-1], t, phi_1) - np.interp(linspace_undistorted_df["t"].to_numpy()[-1], t, phi_1)
phase_2 = np.interp(linspace_df["t"].to_numpy()[-1], t, phi_2) - np.interp(linspace_undistorted_df["t"].to_numpy()[-1], t, phi_2)

U_virtual = expm(1j * 0.5 * (phase_1 * IZ + phase_2 * ZI)) @ expm(-1j * 0.25 * linspace_df["v"].to_numpy()[-1] * (IZ - ZI)) @ U_virtual

infidelity = py_ste.unitary_gate_infidelity(U_physical, U_virtual)
no_z_infidelity = py_ste.unitary_gate_infidelity(U_physical, U_no_z)

print(f"Infidelity is {infidelity} (compared to {no_z_infidelity})")