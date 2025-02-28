import pandas as pd
from fun import *

fileName='Data/241212_Dataset_Leopard24_IMU.csv'
data = pd.read_csv(fileName, sep=",")

fig=plot_angle_vs_height(data.loc[:,'ElbowAngle'].values, data.loc[:, "RightHandZ"].values, data.loc[:, "Trial_ID"].values, 'ElbowAngle')
fig.show()