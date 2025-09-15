import os

import numpy as np
import pandas as pd
import py_ste

from scipy.linalg import expm

from src.setup.get_dir import DATA_DIR
from scripts.data_collection.tasks.pauli_strings import IZ, ZI, IX, XI, XX, YY, ZZ

data_dir = os.path.join(DATA_DIR, "figure_4")
os.makedirs(data_dir, exist_ok=True)
figure_2_data_dir = os.path.join(DATA_DIR, "figure_2")

if not os.path.exists(os.path.join(figure_2_data_dir, "data.csv")):
    print("Data for figure 2 not found. Collecting data for figure 2:")
    from scripts.data_collection import figure_2
    print("Data for figure 2 collected.")

frequency_1 = 2 * np.pi * 18 / 0.03
frequency_2 = frequency_1 - 2 * np.pi
H0 = (0.5 * frequency_1 * IZ + 0.5 * frequency_2 * ZI)

IQ_scale = 2 * np.pi * 0.0026 / 0.03
J_scale = 2 * np.pi * 0.010 / 0.03
virtual_z_evolver: py_ste.evolvers.DenseUnitaryEvolver \
    = py_ste.get_unitary_evolver(H0, [IX + XI, XX + YY + ZZ])
physical_z_evolver: py_ste.evolvers.DenseUnitaryEvolver \
    = py_ste.get_unitary_evolver(H0, [IX + XI, XX + YY + ZZ, IZ, ZI])

df = pd.read_csv(os.path.join(figure_2_data_dir, "data.csv"), index_col=False)

for name, group in df.groupby("name"):
    zero_amplitude_group = group[group["amplitude"] == 0]
    t = zero_amplitude_group["f"].to_numpy()
    T = t[-1] - t[0]
    dt = t[1] - t[0]
    I = IQ_scale * zero_amplitude_group["I"].to_numpy()
    Q = IQ_scale * zero_amplitude_group["Q"].to_numpy()
    g = np.cos(frequency_1 * t) * I + np.sin(frequency_1 * t) * Q
    J = J_scale * zero_amplitude_group["J"].to_numpy()

    if name != "tanh2":
        max_v = []
        infidelities = []
    for amp, sub_group in group.groupby("amplitude"):
        # v corresponds to upper case V in the paper and dv to lower case v in
        #   the paper
        v_diff = np.diff(sub_group["v"].to_numpy())
        f = sub_group["f"].to_numpy()
        dv = v_diff / dt
        dv = np.concatenate([[0], dv])
        physical_ctrl_amp = np.stack([g, J, 0.25 * dv, -0.25 * dv], axis=-1)
        physical_ctrl_amp = physical_ctrl_amp.astype(complex)
        U_physical = physical_z_evolver.get_evolution(
                        physical_ctrl_amp,
                        dt)
        no_z_ctrl_amp = np.stack([g, J], axis=-1)
        no_z_ctrl_amp = no_z_ctrl_amp.astype(complex)
        U_no_z = virtual_z_evolver.get_evolution(
                    no_z_ctrl_amp,
                    dt)
        U_no_z = expm(-1j * 0.25 * sub_group["v"].to_numpy()[-1] * (IZ-ZI)) @ U_no_z
        I_new = IQ_scale * sub_group["I"].to_numpy()
        Q_new = IQ_scale * sub_group["Q"].to_numpy()
        J_new = J_scale * sub_group["J"].to_numpy()
        t = sub_group["t"].to_numpy()
        t_samples = np.linspace(t.min(), t.max(), len(t))
        I_new = np.interp(t_samples, t, I_new)
        Q_new = np.interp(t_samples, t, Q_new)
        J_new = np.interp(t_samples, t, J_new)
        g_new = np.cos(frequency_1 * t_samples) * I_new + np.sin(frequency_1 * t_samples) * Q_new
        virtual_ctrl_amp = np.stack([g_new, J_new], axis=-1)
        virtual_ctrl_amp = virtual_ctrl_amp.astype(complex)
        U_virtual = virtual_z_evolver.get_evolution(
                        virtual_ctrl_amp,
                        t_samples[1] - t_samples[0])

        U_virtual = expm(1j * H0 * (t_samples[-1] - t_samples[0] - T)) @ expm(-1j * 0.25 * sub_group["v"].to_numpy()[-1] * (IZ - ZI)) @ U_virtual

        infidelity = py_ste.unitary_gate_infidelity(U_physical, U_virtual)
        no_z_infidelity = py_ste.unitary_gate_infidelity(U_physical, U_no_z)

        print(f"Infidelity for {name} with amplitude {amp}: {infidelity} (compared to {no_z_infidelity})")
        if amp == 0:
            continue
        max_v.append(np.max(np.abs(dv)))
        infidelities.append(infidelity)
    if name != "tanh":
        max_v = 30 * np.array(max_v) / (2 * np.pi)
        infidelities = np.array(infidelities)
        infidelity_at_half_splitting = np.interp(15, max_v, infidelities)
        print(f"Infidelity at max |v|=15MHz for {name}: {infidelity_at_half_splitting}")
        x = np.log(max_v)
        y = np.log(infidelities)
        n = len(max_v)
        sxy2 = np.sum((x - np.mean(x)) * (y - np.mean(y))) / n
        sx2 = np.sum(np.square(x - np.mean(x))) / n
        gradient = sxy2 / sx2

        infidelity_df = pd.DataFrame(data={"max_v": max_v, "infidelity": infidelities})
        infidelity_df.to_csv(os.path.join(data_dir, f"infidelity_{name}.csv"), index=False)
        with open(os.path.join(data_dir, f"infidelity_{name}_gradient.txt"), "w") as file:
            file.write(f"{gradient}")
        print(f"Gradient for {name} is {gradient}")