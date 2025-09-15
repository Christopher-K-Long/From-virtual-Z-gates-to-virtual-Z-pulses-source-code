import os
import pickle as pkl

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm

from src.setup.get_dir import DATA_DIR

data_dir = os.path.join(DATA_DIR, "figure_3")
os.makedirs(data_dir, exist_ok=True)

energy_scale = 50
width = 0.05
amplitude = 3
resonator_frequency = 2 * np.pi * energy_scale * 8

def get_f(coefficients, b):
    b = b
    powers = (4 + np.arange(len(coefficients)))
    A = np.sum(coefficients * np.power(b, powers))
    B = np.sum(coefficients * powers * np.power(b, powers - 1))
    coeff2 = (-3 * A + b * (B - 3) + 3) / b**2
    coeff3 = (2 * A - b * (B - 2) - 2) / b**3
    f = lambda x: x + coeff2 * x**2 + coeff3 * x**3 + np.sum(np.expand_dims(coefficients, axis=0) * np.power.outer(x, powers), axis=-1)
    df = lambda x: 1 + 2 * coeff2 * x + 3 * coeff3 * x**2 + np.sum(np.expand_dims(coefficients * powers, axis=0) * np.power.outer(x, powers - 1), axis=-1)

    return f, df

# v corresponds to upper case V in the paper and dv to lower case v in the paper
def v(x):
    return amplitude * 0.5 * np.pi * (1 + np.tanh((x - 0.5) / width))

def dv(x):
    return amplitude * 0.5 * np.pi / width * np.cosh((x - 0.5) / width) ** -2

def delta_frequency_old(x):
    return 2 * np.pi * energy_scale * (0.1 - 0.1 * np.exp(-np.square(x - 0.5) / 0.2 ** 2))

def comp_frequency_old(x):
    return 2 * np.pi * energy_scale * (12 + 0.1 * np.exp(-np.square(x - 0.5) / 0.2 ** 2))

def construct_t_specific_patches(f, n_lines, eps=1e-1):
    pbar = tqdm(total=n_lines * np.round(1 / eps), desc="Constructing set of times used to solve recursion relations")
    init_ts = np.linspace(0, eps, n_lines + 1)[1:]
    t = [np.array(init_ts)]
    pbar.update(n_lines)
    while np.max(t) < 1:
        t.append(f(t[-1]))
        pbar.total = len(t) * n_lines * np.round(1 / np.max(t))
        pbar.update(n_lines)
    pbar.close()
    return np.array(t)

def get_delta_phi(t):
    phi = np.zeros_like(t)
    phi[0, 0] = 0
    for j in range(1, len(t)):
        phi[j, 0] = phi[j-1, 0] - v(t[j, 0])
    for i in tqdm(range(1, t.shape[1]), desc="Solving recursion relations for delta phi"):
        t_flat = t[:, :i].flatten()
        phi_flat = phi[:, :i].flatten()
        indices = np.argsort(t_flat)
        t_flat = t_flat[indices]
        phi_flat = phi_flat[indices]
        phi[0, i] = np.interp(t[0, i], t_flat, phi_flat)
        for j in range(1, len(t)):
            phi[j, i] = phi[j - 1, i] - v(t[j, i])
    return phi

def get_comp_dphi(t, f, df):
    dphi = np.zeros_like(t)
    dphi[0, 0] = 2 * (comp_frequency_old(0) - resonator_frequency)
    for j in range(1, len(t)):
        dphi[j, 0] = ((dphi[j - 1, 0] - comp_frequency(t[j-1, 0], f, df)) / df(t[j-1, 0]) + comp_frequency_old(t[j, 0]))[0]
    for i in tqdm(range(1, t.shape[1]), desc="Solving recursion relations for complementary phi"):
        t_flat = t[:, :i].flatten()
        dphi_flat = dphi[:, :i].flatten()
        indices = np.argsort(t_flat)
        t_flat = t_flat[indices]
        dphi_flat = dphi_flat[indices]
        dphi[0, i] = np.interp(t[0, i], t_flat, dphi_flat)
        for j in range(1, len(t)):
            dphi[j, i] = ((dphi[j - 1, i] - comp_frequency(t[j-1, i], f, df)) / df(t[j-1, i]) + comp_frequency_old(t[j, i]))[0]
    return dphi

def delta_frequency(t, f, df):
    return df(t) * (delta_frequency_old(f(t)) + dv(f(t)))

def comp_frequency(t, f, df):
    comp_frequency_old_values = comp_frequency_old(f(t))
    delta_frequency_old_values = delta_frequency_old(f(t))
    g = df(t) * (1 / (0.5 * (comp_frequency_old_values + delta_frequency_old_values) - resonator_frequency)
                +1 / (0.5 * (comp_frequency_old_values - delta_frequency_old_values) - resonator_frequency))
    return (resonator_frequency * g + 1 + np.sqrt(1 + np.square(0.5 * g * delta_frequency(t, f, df)))) / (0.5 * g)

def flatten_and_order(t, phi, comp_phi):
    t = t.flatten()
    phi = phi.flatten()
    comp_phi = comp_phi.flatten()
    indices = np.argsort(t)
    t = t[indices]
    phi = phi[indices]
    comp_phi = comp_phi[indices]
    return t, phi, comp_phi

def relu(x):
    return (x+np.abs(x))/2

def cost(x):
    lagrange_multiplier1 = x[0]
    lagrange_multiplier3 = x[1]
    lagrange_multiplier4 = x[2]
    b = x[3]
    coefficients = x[4:]
    f, df = get_f(coefficients, b)
    t = np.linspace(0, b, 10000)
    delta_frequency_values = delta_frequency(t, f, df)
    comp_frequency_values = comp_frequency(t, f, df)
    frequency_1 = 0.5 * (delta_frequency_values + comp_frequency_values)
    frequency_2 = 0.5 * (comp_frequency_values - delta_frequency_values)
    delta_frequency_old_values = delta_frequency_old(t)
    comp_frequency_old_values = comp_frequency_old(t)
    frequency_1_old = 0.5 * (delta_frequency_old_values + comp_frequency_old_values)
    frequency_2_old = 0.5 * (comp_frequency_old_values - delta_frequency_old_values)
    return np.sum(np.sqrt(np.square(frequency_1-frequency_1_old) + np.square(frequency_2-frequency_2_old))) / (energy_scale * len(t)) \
           + lagrange_multiplier1 * np.sum(relu(t - f(t)))\
           + lagrange_multiplier3 * np.sum(relu(frequency_1_old[0] - frequency_1)) \
           + lagrange_multiplier4 * np.sum(relu(frequency_2_old[0] - frequency_2))

def fft(func):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(func, axes=-1), axis=-1), axes=-1)

def ifft(func):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(func, axes=-1), axis=-1), axes=-1)

def fftfreq(x):
    return 2*np.pi*np.fft.fftshift(np.fft.fftfreq(len(x), x[1]-x[0]))

def get_IQ(g_samples, t_samples, frequency, pad_multiplier=1):
    right_pad = len(g_samples)*(pad_multiplier-1)
    left_pad = len(g_samples)*(pad_multiplier)
    g_samples = np.concatenate([[0]*left_pad, g_samples, [0]*right_pad])
    t_samples = np.concatenate([np.flip(t_samples[0]-(t_samples[1]-t_samples[0])*np.arange(1, left_pad+1)),
                                t_samples,
                                t_samples[-1]+(t_samples[1]-t_samples[0])*np.arange(1, right_pad+1)])
    fft_g = fft(g_samples)
    k = fftfreq(frequency*t_samples)
    fft_IQ_pre_transform = np.array([np.interp(k + 1, k, fft_g)*(k + 1 >= 0),
                                     np.interp(k - 1, k, fft_g)*(1 - k >= 0)])
    fft_IQ = np.array([[1,   1],
                       [1j, -1j]]) @ fft_IQ_pre_transform
    IQ = ifft(fft_IQ)
    return IQ[:, left_pad : IQ.shape[1]-right_pad]


print("Optimising dilation...")
n_var = 8
result = minimize(cost, [1.1] * 3 + [0.9] + [0.] * n_var, bounds=[(1, None)] * 3 + [(None, None)] + [(None, None)] * n_var, options={"disp": True}, method="COBYLA")
print("Optimisation done.")

with open(os.path.join(data_dir, "optimization_result.pkl"), "wb") as file:
    pkl.dump(result, file, pkl.HIGHEST_PROTOCOL)

f, df = get_f(result.x[4:], result.x[3])
t = np.linspace(0, result.x[3], 10000)
f_undilated, df_undilated = get_f([0] * n_var, 1)

print("Solving for distorted qubit frequencies...")
linspace_df = pd.DataFrame(data={"t": t * energy_scale,
                                 "f": f(t) * energy_scale,
                                 "df": df(t),
                                 "v": v(t),
                                 "dv": dv(t) / (energy_scale * 2 * np.pi),
                                 "distorted_v": df(t) * v(f(t)),
                                 "frequency_1_old": 0.5 * (delta_frequency_old(t) + comp_frequency_old(t)) / (energy_scale * 2 * np.pi),
                                 "frequency_2_old": 0.5 * (comp_frequency_old(t) - delta_frequency_old(t)) / (energy_scale * 2 * np.pi),
                                 "frequency_1": 0.5 * (delta_frequency(t, f_undilated, df_undilated) + comp_frequency(t, f_undilated, df_undilated)) / (energy_scale * 2 * np.pi),
                                 "frequency_2": 0.5 * (comp_frequency(t, f_undilated, df_undilated) - delta_frequency(t, f_undilated, df_undilated)) / (energy_scale * 2 * np.pi),
                                 "frequency_1_distorted": 0.5 * (delta_frequency(t, f, df) + comp_frequency(t, f, df)) / (energy_scale * 2 * np.pi),
                                 "frequency_2_distorted": 0.5 * (comp_frequency(t, f, df) - delta_frequency(t, f, df)) / (energy_scale * 2 * np.pi)})
print("Solving for qubit frequencies done.")

t = np.linspace(0, 1, 100000)
linspace_undistorted_df = pd.DataFrame(data={"t": t * energy_scale,
                                     "f": f(t) * energy_scale,
                                     "df": df(t),
                                     "v": v(t),
                                     "dv": dv(t) / (energy_scale * 2 * np.pi),
                                     "frequency_1_old": 0.5 * (delta_frequency_old(t) + comp_frequency_old(t)) / (energy_scale * 2 * np.pi),
                                     "frequency_2_old": 0.5 * (comp_frequency_old(t) - delta_frequency_old(t)) / (energy_scale * 2 * np.pi),
                                     "frequency_1": 0.5 * (delta_frequency(t, f_undilated, df_undilated) + comp_frequency(t, f_undilated, df_undilated)) / (energy_scale * 2 * np.pi),
                                     "frequency_2": 0.5 * (comp_frequency(t, f_undilated, df_undilated) - delta_frequency(t, f_undilated, df_undilated)) / (energy_scale * 2 * np.pi)})

t = construct_t_specific_patches(f, 1000)
delta_phi = get_delta_phi(t)
comp_dphi = get_comp_dphi(t, f, df)

t, delta_phi, comp_dphi = flatten_and_order(t, delta_phi, comp_dphi)
delta_dphi = np.concatenate([[0], np.diff(delta_phi) / np.diff(t)])
i = np.argmax(t > 1) + 1
if i == 0:
    i = -1
delta_phi = delta_phi[:i]
delta_dphi = delta_dphi[:i]
t = t[:i]
comp_dphi = comp_dphi[:i]
comp_phi = cumulative_trapezoid(comp_dphi, t, initial=0)

phi_1 = 0.5 * (delta_phi + comp_phi)
phi_2 = 0.5 * (comp_phi - delta_phi)

recursive_df = pd.DataFrame(data={"t": t * energy_scale,
                                  "dphi_1": 0.5 * (delta_dphi + comp_dphi) / (energy_scale * 2 * np.pi),
                                  "dphi_2": 0.5 * (comp_dphi - delta_dphi) / (energy_scale * 2 * np.pi),
                                  "phi_1": phi_1,
                                  "phi_2": phi_2})

print("Saving rotating frame frequencies...")
recursive_df.to_csv(os.path.join(data_dir, "recursive_data.csv"), index=False)
print("Saving rotating frame frequencies done.")

frequency = 2*np.pi*energy_scale*linspace_df["frequency_1_old"][0]
pulse = lambda x: np.exp(-np.square(x-0.5) / 0.15**2)
g = lambda x: np.cos(frequency*x)*pulse(x)

phi = phi_1

print("Distorting IQ pulse...")
x = np.linspace(phi.min(), phi.max(), 10000)
t_samples = np.interp(x, phi, t)
g_samples = g(t_samples)
IQ = get_IQ(g_samples, x, 1, 100)
f_values = f(t_samples)
new_IQ = df(t_samples) * np.einsum("ijt,jt->it",
                                    np.array([[np.cos(0.5 * v(f_values)), -np.sin(0.5 * v(f_values))],
                                              [np.sin(0.5 * v(f_values)),  np.cos(0.5 * v(f_values))]]),
                                    np.array([np.interp(f_values, t_samples, IQ[0]),
                                              np.interp(f_values, t_samples, IQ[1])]))
new_t = np.linspace(t.min(), t.max(), 10000)
new_IQ = np.array([np.interp(new_t, t_samples, new_IQ[0]),
                    np.interp(new_t, t_samples, new_IQ[1])])
new_phi = np.interp(new_t, t, phi)
new_g = np.cos(new_phi) * new_IQ[0] + np.sin(new_phi) * new_IQ[1]
final_IQ = get_IQ(new_g, new_t, frequency, 100)
final_IQ = [np.interp(linspace_df["t"].to_numpy(), new_t * energy_scale, final_IQ[0]),
            np.interp(linspace_df["t"].to_numpy(), new_t * energy_scale, final_IQ[1])]
linspace_df["I_old"] = pulse(linspace_df["t"] / energy_scale)
linspace_df["Q_old"] = 0 * linspace_df["I_old"]
linspace_undistorted_df["I_old"] = pulse(linspace_undistorted_df["t"] / energy_scale)
linspace_undistorted_df["Q_old"] = 0 * linspace_undistorted_df["I_old"]
linspace_df["I_new"] = final_IQ[0].real
linspace_df["Q_new"] = final_IQ[1].real
print("Distorting IQ pulse done.")

print("Saving data...")
linspace_df.to_csv(os.path.join(data_dir, "linspace_data.csv"), index=False)
linspace_undistorted_df.to_csv(os.path.join(data_dir, "linspace_undistorted_data.csv"), index=False)
print("Saving data done.")