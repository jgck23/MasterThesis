import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.signal import convolve

def main():
    folder_path = '/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/Xsens Data/241113_Leopard24'
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]
    file_names = sorted(file_names)
    print(f'These files will be vertically concatenated in the following order: {file_names}')
    NN = []
    # load the number of frames for each trial from the pressure sensor and check if trial count matches
    N_frames = np.loadtxt('Data/Foot Sensor Force Data/241113_Leopard24_N_frames_FSensor.csv', delimiter=',')
    if len(N_frames) != len(file_names):
        raise ValueError("The number of trials in the sensor data and the Xsens data do not match.")
    
    print(len(file_names))
    # iterate over all files in the folder and append the data to the list, also cut the xsens data accordingly
    for i in range(len(file_names)):
        file_path = os.path.join(folder_path, file_names[i])
        data = load_data(file_path)
        if data.shape[0] < N_frames[i]:
            raise ValueError(f'Number of Xsens frames from file: {file_names[i]} is smaller than the number of sensor frames. Manually remove the sensor data and the xsens data.')
        data = data[:int(N_frames[i])] # cut the xsens data to match the sensor data
        data = np.array(data, dtype=float)
        data = smooth_data(data, f_cutoff=5, recording_frequency=60)
        NN.append(data)
    
    NNarray = np.vstack(NN)

    # save the data to a csv file
    df = pd.DataFrame(NNarray)
    df.to_csv('Data/Xsens Data/241113_Leopard24_Xsens.csv', index=False, header=False)

def load_data(file_path):
    # read the data from the given csv file and store it in a list of arrays
    trial = []
    with open(file_path, 'r') as file:
        for line in file:
            trial.append(line.strip().split(','))
        array = np.array(trial)
              
    return array         

def smooth_data(data, f_cutoff=1, recording_frequency=60):
    sigma = recording_frequency / (2 * np.pi * f_cutoff)
    # create a Gaussian filter
    filter_length = max(3, int(6 * sigma - 1))
    gauss_filter = gaussian(
        filter_length, std=sigma
    )  # length: 6 * sigma - 1, standard deviation: sigma
    gauss_filter /= gauss_filter.sum()  # normalization

    # Smooth signals
    smoothed_data = np.zeros_like(data)  # empty array with the same shape as the data
    for i in range(data.shape[1]):
        signal_data = data[:, i]  # smooth each column of the data
        pad_width = len(gauss_filter) // 2  # Pad the data to avoid edge effects
        padded_signal = np.pad(
            signal_data, pad_width=pad_width, mode="edge"
        )  # edge padding, uses the value of the nearest edge
        smoothed_signal = convolve(
            padded_signal, gauss_filter, mode="same"
        )  # Convolve the signal with the Gaussian filter, mode='same' returns the same length as the input
        smoothed_data[:, i] = smoothed_signal[
            pad_width:-pad_width
        ]  # Trim the padded output to match original length

    '''plt.plot(data[:, 2], label='Original')
    plt.plot(smoothed_data[:, 2], label='Smoothed')
    plt.legend()
    plt.show()'''

    return smoothed_data       
    

if __name__ == "__main__":
    main()