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
########################################

folder_path_Xsens = ['/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/Xsens Data/241113_Leopard24','/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/Xsens Data/241121_Leopard24','/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/Xsens Data/241212_Leopard24']
folder_path_Tekscan = ['/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/Foot Sensor Force Data/241113_Leopard24','/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/Foot Sensor Force Data/241121_Leopard24','/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/Foot Sensor Force Data/241212_Leopard24']
output_name = '241113_241121_241212_combined.csv'

#!!!!!!!Dont change anything below this line!!!!!!!

tek_data, tek_frames = sensor_data.main(folder_path_Tekscan)
xs_data, xs_frames = xsens_data.main(folder_path_Xsens)

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

#convert to pandas dataframe and save as csv
df = pd.DataFrame(ml_stack)
df.to_csv(output_name, index=False, header=False)