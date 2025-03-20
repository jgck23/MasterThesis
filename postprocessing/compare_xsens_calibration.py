import os
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from natsort import natsorted

directory = '/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/Xsens Data/250312_Pferd12_Cal'

file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(".csv")]
file_names = natsorted(file_names)

fig = go.Figure()
for i in range(len(file_names)):
    file_path = os.path.join(directory, file_names[i])
    data = pd.read_csv(file_path)
    fig.add_trace(go.Scatter( y=data['ElbowAngle'].values[0:120], mode='lines', name=file_names[i]))
fig.show()