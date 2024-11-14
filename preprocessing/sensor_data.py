import os
import pandas as pd
import numpy as np

def main():
    folder_path = '/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/Foot Sensor Force Data/241113_Leopard24'
    # file names are not in order
    # sensor data already has cut off for minimum values (S.Helmstetter)
    # if data smoothing is needed, it can be done here per trial
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    NN = []
    for i in range(len(file_names)):
        file_path = os.path.join(folder_path, file_names[0])
        data = load_data(file_path)
        data= np.insert(data, 0, i+1, axis=1)
        NN.append(data)
    
    NNarray = np.vstack(NN)
    zero_column = [i for i in range(NNarray.shape[1]) if np.all(NNarray[:, i] == 0)]
    print(f"Zero columns: {zero_column}")
    NNarray = np.delete(NNarray, zero_column, axis=1)

    print(NN.shape)

def load_data(file_path):
    # read the data from the given csv file and store it in a list of arrays
    sensordata = []
    currentframe = []
    start_there = False
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('ASCII_DATA @@'):
                start_there = True
                continue

            if not start_there:
                continue
            else:
                if line.startswith('Frame'):
                    if currentframe:
                        frame_array = np.array(currentframe, dtype=int)
                        sensordata.append(frame_array)
                        currentframe = []
                elif line and line.startswith(('0','B')):
                    # Process the current row, converting 'B' to 9999 to filter it out later
                    row = [9999 if val == 'B' else int(val) for val in line.split(',')]
                    currentframe.append(row)
    
    if currentframe:
        frame_array = np.array(currentframe, dtype=int)
        sensordata.append(frame_array)

    #delete unwanted rows from each frame
    for _ in range(len(sensordata)):
        sensordata[_] = sensordata[_][12:39,:] # values have to be adjusted according to the data by hand

    #flatten the data and convert it to an array
    list = []
    for i in range(len(sensordata)):
        list.append(sensordata[i].flatten())
        array = np.array(list)
    
    # delete all columns previously filled with 9999, from sensor: 'B'
    columns_to_delete = [i for i in range(array.shape[1]) if np.all(array[:, i] == 9999)]
    new_array = np.delete(array, columns_to_delete, axis=1)
              
    return new_array                
    

if __name__ == "__main__":
    main()