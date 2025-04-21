import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from fun import *

#fileName='Data/241212_Dataset_Leopard24_IMU.csv'
#fileName='Data/250312_Dataset_Pferd12.csv'
fileName='Data/250318_Dataset_Eule3.csv'
data = pd.read_csv(fileName, sep=",")

#fig=plot_angle_vs_height(data.loc[:,'ElbowAngle'].values, data.loc[:, "RightHandZ"].values, data.loc[:, "Trial_ID"].values, 'ElbowAngle')
#fig.show()

def plot_trial_duration(data):
    #selection = (data.loc[:, "RightHandZ"] < 500)
    #data = data.loc[selection]
    trial_ids = data.loc[:, "Trial_ID"].values
    
    fig = go.Figure()
    unique_trials = np.unique(trial_ids)
    
    lengths = []
    height = []
    for trial_id in unique_trials:
        mask = trial_ids == trial_id
        lengths.append(trial_ids[mask].shape[0])
        height.append(np.mean(data.loc[mask, "RightHandZ"].values))
    fig.add_trace(go.Histogram2d(
        x=lengths,
        y=height,
        xbins=dict(size=20),
        ybins=dict(size=20),
        ))
    fig.update_layout(
        xaxis_title="Drilling Duration", 
        yaxis_title="Height", 
        title=f"Histogram of Drilling Duration", #Plot der Höhe gegen den {zielvariable}
        title_font=dict(size=30),
        xaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
        yaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
    )

    average = np.mean(lengths)
    print(f"Average drilling duration: {average}")
    return fig

def plot_trial_duration_express(data):
    trial_ids = data.loc[:, "Trial_ID"].values
    unique_trials = np.unique(trial_ids)
    lengths = []
    height = []
    for trial_id in unique_trials:
        mask = trial_ids == trial_id
        lengths.append(trial_ids[mask].shape[0])
        height.append(np.mean(data.loc[mask, "RightHandZ"].values))
    
    fig = px.density_heatmap(x=lengths, y=height, nbinsx=30, nbinsy=30, marginal_x="histogram", marginal_y="histogram")
    fig.update_layout(
        xaxis_title="Drilling Duration", 
        yaxis_title="Height", 
        title=f"Histogram of Drilling Duration", #Plot der Höhe gegen den {zielvariable}
        title_font=dict(size=30),
        xaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
        yaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
    )
    return fig

def plot_trial_duration_1dhist(data):
    #selection = (data.loc[:, "RightHandZ"] < 500)
    #data = data.loc[selection]
    trial_ids = data.loc[:, "Trial_ID"].values
    
    fig = go.Figure()
    unique_trials = np.unique(trial_ids)
    
    lengths = []
    for trial_id in unique_trials:
        mask = trial_ids == trial_id
        lengths.append(trial_ids[mask].shape[0])
    fig.add_trace(go.Histogram(
        x=lengths,
        nbinsx=30,
        marker=dict(color='blue'),
    ))
    fig.update_layout(
        xaxis_title="Drilling Duration",
        yaxis_title="Count",
        title=f"Histogram of Drilling Duration", #Plot der Höhe gegen den {zielvariable}
        title_font=dict(size=30),
        xaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
        yaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
    )

    average = np.mean(lengths)
    print(f"Average drilling duration: {average}")
    return fig

fig = plot_trial_duration_1dhist(data)
fig.show()