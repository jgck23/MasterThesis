import pandas as pd
import sensor_data
import xsens_data
import numpy as np

############IMPORTANT NOTICE############
#all files in the respective folders 
#must be in the same order, the same 
#name is not required. This means first 
#recording at top and last recording at
#bottom in both folders when sorted by 
#name. If provided a list of folders,
#the order of the folders must be the
#same as the order of the recordings.
#Tekscan data aswell as Xsens data are 
#both filtered with 1Hz lowpass filter 
#edge padding. Change the frequencey 
#in the respective files if needed.

#Also change the sector of the sensor in sensor_data.py in line 93 to your specific attachment of the sensor to the power tool handle.
########################################

folder_path_Xsens = ['/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/Xsens Data/250318_Eule3']
folder_path_Tekscan = ['/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/Foot Sensor Force Data/250318_Eule3']
output_name = '250318_Dataset_Eule3.csv'

#!!!!!!!Dont change anything below this line!!!!!!!

tek_data, tek_frames = sensor_data.main(folder_path_Tekscan)
xs_data, xs_frames, header = xsens_data.main(folder_path_Xsens)

ml = []
comb = []
if len(tek_frames) != len(xs_frames):
    raise ValueError('Count of trials does not match')

for i in range(len(tek_frames)): # iterate over all trials and horizontally concatenate the data
    if tek_frames[i] == xs_frames[i]:
        comb = np.hstack([tek_data[i], xs_data[i]])
        ml.append(comb)
    elif tek_frames[i] < xs_frames[i]:
        comb = np.hstack([tek_data[i][0:tek_frames[i],:], xs_data[i][0:tek_frames[i],:]])
        if xs_frames[i] - tek_frames[i] > 5:
            print(f"Trial {i+1} has different frame counts. Tekscan: {tek_frames[i]}, Xsens: {xs_frames[i]}")
        ml.append(comb)
    elif tek_frames[i] > xs_frames[i]:
        comb = np.hstack([tek_data[i][0:xs_frames[i],:], xs_data[i][0:xs_frames[i],:]])
        if tek_frames[i] - xs_frames[i] > 5:
            print(f"Trial {i+1} has different frame counts. Tekscan: {tek_frames[i]}, Xsens: {xs_frames[i]}")
        ml.append(comb)

# delete feature columns filled with zeros, there is no information
ml_stack = np.vstack(ml)
zero_column = [i for i in range(ml_stack.shape[1]) if np.all(ml_stack[:, i] == 0)]
# print(f"Zero columns: {zero_column}")
ml_stack = np.delete(ml_stack, zero_column, axis=1)

#add header to the data
xsens_header_count=len(header)
header_list = ['Trial_ID'] + [f'sensor{i+1}' for i in range(ml_stack.shape[1] -4 -xsens_header_count)] + ['avg_load', 'total_load', 'active_sensors']
header_list = header_list + header

#convert to pandas dataframe and save as csv
df = pd.DataFrame(ml_stack)
df.to_csv(output_name, index=False, header=header_list)