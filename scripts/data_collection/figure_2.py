import os

import numpy as np
import pandas as pd

from tqdm import tqdm

from src.setup.get_dir import DATA_DIR

data_dir = os.path.join(DATA_DIR, "figure_2")
os.makedirs(data_dir, exist_ok=True)

f = np.linspace(0, 1, 100000)
frequency = 2 * np.pi

def get_t(v):
    return (v(f) + frequency * f) / frequency

def get_J(pulse, dv):
    return pulse(f) / (1 + dv(f) / frequency)

def get_IQ(IQ, dv):
    rotation = np.array([[np.cos(0.5*v(f)), -np.sin(0.5*v(f))],
                         [np.sin(0.5*v(f)),  np.cos(0.5*v(f))]])
    IQ_values = np.einsum("ijt,jt->it", rotation, IQ(f))
    return IQ_values / np.expand_dims(1 + dv(f)/frequency, axis=0)

width1 = 0.15
width2 = 0.1

# vs corresponds to upper case V in the paper and dvs to lower case v in the
#   paper
vs = [
    lambda amplitude: (lambda x: amplitude * 0.6 * np.exp(-np.square(x-0.5) / width1**2)),
    lambda amplitude: (lambda x: amplitude * (1+np.tanh((x - 0.5) / width2))),
    lambda amplitude: (lambda x: 10 * amplitude * (1+np.tanh((x - 0.5) / width2)))
]

dvs = [
    lambda amplitude: (lambda x: -2 * amplitude * 0.6 * (x - 0.5) * np.exp(-np.square(x - 0.5) / width1**2) / width1**2),
    lambda amplitude: (lambda x: amplitude * np.cosh((x - 0.5) / width2)**-2 / width2),
    lambda amplitude: (lambda x: 10 * amplitude * np.cosh((x - 0.5) / width2)**-2 / width2)
]

dataframe = None

progress_bar = tqdm(total=16, desc="Virtual Z pulses")

for v_getter, dv_getter, name in zip(vs, dvs, ["gaussian", "tanh", "tanh2"]):

    amplitudes = np.linspace(0, 1, 5)
    if name == "tanh":
        amplitudes = np.concatenate([[-0.25], amplitudes])
    
    for amplitude in amplitudes:
        v = v_getter(amplitude)
        dv = dv_getter(amplitude)
        
        pulse = lambda x: np.exp(-np.square(x-0.5) / (width1*2)**2)
        IQ = lambda x: np.stack([pulse(x), np.zeros_like(x)], axis=0)

        I, Q = get_IQ(IQ, dv)
        new_dataframe = pd.DataFrame(data={"name": [name]*len(f),
                                           "amplitude": [amplitude]*len(f),
                                           "f": f,
                                           "t": get_t(v),
                                           "v": v(f),
                                           "J": get_J(pulse, dv),
                                           "I": I,
                                           "Q": Q})
        if dataframe is None:
            dataframe = new_dataframe
        else:
            dataframe = pd.concat([dataframe, new_dataframe])
        progress_bar.update(1)
progress_bar.close()
print("Saving data...")
dataframe.to_csv(os.path.join(data_dir, "data.csv"), index=False)
print("Data saved.")