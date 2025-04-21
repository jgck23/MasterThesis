# this file loads the Xsens data from the csv files, smoothes the data and gives it to comb_forcesensor_xsens.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.signal import convolve
from natsort import natsorted


def main(folder_paths):
    NN_array = []
    N_frames_array = []

    for folder_path in folder_paths:
        file_names = [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".csv")
        ]
        file_names = natsorted(file_names)
        print(
            f"These files will be vertically concatenated in the following order: {file_names}"
        )
        NN = []
        N_frames = []

        # iterate over all files in the folder and append the data to the list, also cut the xsens data accordingly
        for i in range(len(file_names)):
            file_path = os.path.join(folder_path, file_names[i])
            header, data = load_data(file_path)
            # if data.shape[0] < N_frames[i]:
            #    raise ValueError(f'Number of Xsens frames from file: {file_names[i]} is smaller than the number of sensor frames. Manually remove the sensor data and the xsens data.')
            # data = data[:int(N_frames[i])] # cut the xsens data to match the sensor data
            data = np.array(data, dtype=float)
            data = smooth_data(data, f_cutoff=5, recording_frequency=60)
            # data = np.insert(data, 0, i + 1, axis=1)  # add trial count
            N_frames.append(data.shape[0])
            NN.append(data)

        NN_array.extend(NN)
        N_frames_array.extend(N_frames)

        # NNarray = np.vstack(NN)

        # save the data to a csv file
        # df1 = pd.DataFrame(NNarray)
        # df2 = pd.DataFrame(N_frames)
        # df.to_csv('Data/Xsens Data/241212_Leopard24_Xsens.csv', index=False, header=False)

    return NN_array, N_frames_array, header


def load_data(file_path):
    # read the data from the given csv file and store it in a list of arrays
    df = pd.read_csv(file_path)
    header = list(df.columns)  # extract the header
    array = df.values  # convert the dataframe to a numpy array

    return header, array


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

    return smoothed_data


if __name__ == "__main__":
    main()
