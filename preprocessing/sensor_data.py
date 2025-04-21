# this file reads the Tekscan .csv files, smoothes the data and gives it to comb_forcesensor_xsens.py
import os
import pandas as pd
import numpy as np
from scipy.signal.windows import gaussian
from scipy.signal import convolve
import matplotlib.pyplot as plt
from natsort import natsorted


def main(folder_paths):
    NN_array = []
    N_frames_array = []
    trials = 0
    for folder_path in folder_paths:
        # sensor data already has cut off for minimum values (S.Helmstetter)
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
        # iterate over all files in the folder and append the data to the list
        for i in range(len(file_names)):
            file_path = os.path.join(folder_path, file_names[i])
            # load the data from the file
            data = load_data(file_path)
            # smooth the data per trial
            data = smooth_data(data, f_cutoff=1, recording_frequency=60)
            data = np.insert(data, 0, i + 1 + trials, axis=1)  # add trial count

            # Add mean of each row and count of activated sensels to the last columns
            row_means = np.round(data.mean(axis=1, keepdims=True), 2)
            total_load = np.sum(data, axis=1, keepdims=True)
            activated_sensels = np.sum(data > 0, axis=1, keepdims=True)
            data = np.hstack((data, row_means))
            data = np.hstack((data, total_load))
            data = np.hstack((data, activated_sensels))

            N_frames.append(data.shape[0])
            NN.append(data)

        trials += len(file_names)

        NN_array.extend(NN)
        N_frames_array.extend(N_frames)

    return NN_array, N_frames_array


def load_data(file_path):
    # read the data from the given csv file and store it in a list of arrays
    sensordata = []
    currentframe = []
    start_there = False
    with open(file_path, "r") as file:
        lines = file.readlines()[:-1]  # Exclude the last line because of '@@'
        for line in lines:  # iterate line for line through the file
            line = line.strip()  # removes leading and trailing whitespaces
            if line.startswith("ASCII_DATA @@"):  # check when the data starts
                start_there = True
                continue

            if not start_there:
                continue
            else:
                if line.startswith("Frame"):
                    if (
                        currentframe
                    ):  # checks if the current frame is not empty, so for the first frame this is skipped
                        frame_array = np.array(currentframe, dtype=int)
                        sensordata.append(frame_array)
                        currentframe = []
                elif line:  # line.startswith(("0", "B"))
                    # Process the current row, converting 'B' to 9999 to filter it out later
                    row = [
                        9999 if val == "B" else int(val) for val in line.split(",")
                    ]  # writes the values of each line split by ',' in 'row' and filters for 'B'
                    currentframe.append(row)  # appends the row to the current frame

    if currentframe:  # appends the last frame to sensordata
        frame_array = np.array(currentframe, dtype=int)
        sensordata.append(frame_array)

    # delete unwanted rows from each frame
    for _ in range(len(sensordata)):
        sensordata[_] = sensordata[_][
            11:38, :
        ]  # values have to be adjusted according to the data by hand, for Leopard24: 12:39, Pferd12 11:38, but it wouldnt matter if different

    # flatten the data and convert it to an array
    list = []
    for i in range(len(sensordata)):
        list.append(sensordata[i].flatten())
        array = np.array(list)

    # delete all columns previously filled with 9999, from sensor: 'B'
    columns_to_delete = [
        i for i in range(array.shape[1]) if np.all(array[:, i] == 9999)
    ]
    new_array = np.delete(array, columns_to_delete, axis=1)

    return new_array


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
