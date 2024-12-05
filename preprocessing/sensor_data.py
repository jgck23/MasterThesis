import os
import pandas as pd
import numpy as np
from scipy.signal.windows import gaussian
from scipy.signal import convolve
import matplotlib.pyplot as plt


def main():
    folder_path = "/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/Foot Sensor Force Data/241113_Leopard24"
    # sensor data already has cut off for minimum values (S.Helmstetter)
    file_names = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".csv")
    ]
    file_names = sorted(file_names)
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
        data = smooth_data(data, f_cutoff=7, recording_frequency=60)
        data = np.insert(data, 0, i + 1, axis=1)  # add trial count

        N_frames.append(data.shape[0])
        NN.append(data)

    # delete columns filled with zeros, no information
    NNarray = np.vstack(NN)
    zero_column = [i for i in range(NNarray.shape[1]) if np.all(NNarray[:, i] == 0)]
    # print(f"Zero columns: {zero_column}")
    NNarray = np.delete(NNarray, zero_column, axis=1)

    #create additional features
    
    # Add the mean of each row to the last column
    row_means = np.round(NNarray.mean(axis=1, keepdims=True), 2)

    # Count all values in a row that are greater than zero and add the count to the last column
    activated_sensels = np.sum(NNarray > 0, axis=1, keepdims=True)

    NNarray = np.hstack((NNarray, row_means))
    NNarray = np.hstack((NNarray, activated_sensels))

    # save the data to a csv file
    df = pd.DataFrame(NNarray)
    df.to_csv(
        "Data/Foot Sensor Force Data/241113_Leopard24_FSensor.csv",
        index=False,
        header=False,
    )
    df = pd.DataFrame(N_frames)
    df.to_csv(
        "Data/Foot Sensor Force Data/241113_Leopard24_N_frames_FSensor.csv",
        index=False,
        header=False,
    )


def load_data(file_path):
    # read the data from the given csv file and store it in a list of arrays
    sensordata = []
    currentframe = []
    start_there = False
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("ASCII_DATA @@"):
                start_there = True
                continue

            if not start_there:
                continue
            else:
                if line.startswith("Frame"):
                    if currentframe:
                        frame_array = np.array(currentframe, dtype=int)
                        sensordata.append(frame_array)
                        currentframe = []
                elif line and line.startswith(("0", "B")):
                    # Process the current row, converting 'B' to 9999 to filter it out later
                    row = [9999 if val == "B" else int(val) for val in line.split(",")]
                    currentframe.append(row)

    if currentframe:
        frame_array = np.array(currentframe, dtype=int)
        sensordata.append(frame_array)

    # delete unwanted rows from each frame
    for _ in range(len(sensordata)):
        sensordata[_] = sensordata[_][
            12:39, :
        ]  # values have to be adjusted according to the data by hand

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

    """plt.plot(data[:, 200], label='Original')
    plt.plot(smoothed_data[:, 200], label='Smoothed')
    plt.legend()
    plt.show()"""

    return smoothed_data


if __name__ == "__main__":
    main()
