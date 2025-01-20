#from fun import load_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.kernel_approximation import Nystroem
import numpy as np
import plotly.graph_objects as go
import colorcet as cc

def plot_y(y_true, y_pred, trial_ids_test):
    # Plot y_pred and y_test as a dot plot for the test set
    # Create a colormap for unique trial IDs
    unique_trials = np.unique(trial_ids_test)
    colors = cc.glasbey

    if len(unique_trials) > len(colors):
        raise ValueError(
            f"Number of unique trials {len(unique_trials)} exceeds the number of available colors {len(colors)}. Not all plots will be drawn. Adjust colormap accordingly."
        )

    fig1 = go.Figure()
    for trial_id, color in zip(unique_trials, colors):
        # Mask to select data points for the current trial
        mask = trial_ids_test == trial_id
        fig1.add_trace(
            go.Scatter(
                x=y_true[mask].flatten(),
                y=y_pred[mask].flatten(),
                mode="markers",
                marker=dict(color=color),
                name=f"Trial {trial_id}",
            )
        )
    fig1.add_trace(
        go.Scatter(
            x=[min(y_true), max(y_true)],
            y=[min(y_true), max(y_true)],
            mode="lines",
            line=dict(color="red"),
            name="Ideale Linie (Ideal Line)",
        )
    )
    fig1.update_traces(opacity=0.75)
    fig1.update_layout(
        xaxis_title="Tatsächliche Werte (True Values)",
        yaxis_title="Vorhergesagte Werte (Predicted Values)",
        title="Plot der tatsächlichen und vorhergesagten Werte (Dot Plot of True and Predicted Values)",
    )

    # Plot a residual plot
    fig2 = go.Figure()
    for trial_id, color in zip(unique_trials, colors):
        # Mask to select data points for the current trial
        mask = trial_ids_test == trial_id
        fig2.add_trace(
            go.Scatter(
                x=y_true[mask].flatten(),
                y=y_pred[mask].flatten() - y_true[mask].flatten(),
                mode="markers",
                marker=dict(color=color),
                name=f"Trial {trial_id}",
            )
        )
    fig2.add_trace(
        go.Scatter(
            x=[min(y_true), max(y_true)],
            y=[0, 0],
            mode="lines",
            line=dict(color="red"),
            name="Ideale Linie (Ideal Line)",
        )
    )
    fig2.update_traces(opacity=0.75)
    fig2.update_layout(
        xaxis_title="Tatsächliche Werte (True Values)",
        yaxis_title="Residuen (Residuals)",
        title="Plot der Residuen (Residual Plot)",
    )


    # plot y_true and y_pred as a line plot
    fig3 = go.Figure()
    for trial_id, color in zip(unique_trials, colors):
        # Mask to select data points for the current trial
        mask = trial_ids_test == trial_id
        fig3.add_trace(
            go.Scatter(
                x=np.arange(len(y_true))[mask],
                y=y_true[mask].flatten(),
                mode="lines",
                line=dict(color=color, dash="solid", width=3),
                name=f"Trial {trial_id} - Tatsächliche Werte (True Values)",
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=np.arange(len(y_pred))[mask],
                y=y_pred[mask].flatten(),
                mode="lines",
                line=dict(color=color, dash="dash", width=1),
                name=f"Trial {trial_id} - Vorhergesagte Werte (Predicted Values)",
            )
        )
    fig3.update_layout(
        xaxis_title="Datenpunkte (Data Points)",
        yaxis_title="Werte (Values)",
        title="Plot der tatsächlichen und vorhergesagten Werte (Line Plot of True and Predicted Values)",
    )

    return fig1, fig2, fig3
    
# Load the data
#data, name_mat = load_data()
#fileName='Data/241113_Dataset_Leopard24.csv'
fileName='Data/241212_Dataset_Leopard24_IMU.csv'
data = pd.read_csv(fileName, sep=',')

decrease_trials = True
decrease_trials_size = 0.3
if decrease_trials:
        trial_ids = data.loc[:, "Trial_ID"].values
        unique_trials = np.unique(trial_ids)
        n_trials = unique_trials.size
        n_trials = round(decrease_trials_size * n_trials)
        random_trials = np.random.choice(unique_trials, n_trials, replace=False) # set random seed !!!!!
        data = data[data["Trial_ID"].isin(random_trials)]

# Split the data into features and target
X = data.loc[:, "sensor1":"active_sensors"].values
y = data.loc[:, "ElbowAngle"].values
trial_ids = data.iloc[:, 0].values

# Initialize GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Split the data
for train_index, test_index in gss.split(X, y, groups=trial_ids):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    trial_ids_train = data.iloc[train_index, 0].values
    trial_ids_test = data.iloc[test_index, 0].values

# Standardize the data
scaler_x = MinMaxScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

# Define the kernel
#kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
#kernel = RBF(length_scale=1.0)
# Define the exponential kernel

# Approximation
#nystroem = Nystroem(kernel=RBF(), n_components=100)
#X_train_transformed = nystroem.fit_transform(X_train)

gpr = GaussianProcessRegressor(kernel=Matern(nu=5/2, length_scale=3.0, length_scale_bounds='fixed'))
gpr.fit(X_train, y_train) 

# Predict on the test data
y_pred, sigma = gpr.predict(X_test, return_std=True)
testing_rmse = root_mean_squared_error(y_test, y_pred)
training_rmse=root_mean_squared_error(y_train, gpr.predict(X_train))
training_r2=r2_score(y_train, gpr.predict(X_train))
testing_r2=r2_score(y_test, y_pred)
print("training rmse:", training_rmse)
print("testing rmse:", testing_rmse)
print("training r2:", training_r2)
print("testing r2:", testing_r2)

# Plot y_pred and y_test as a dot plot for the test set
plot1, plot2, plot3 = plot_y(y_test, y_pred, trial_ids_test)
plot1.show()
plot2.show()
plot3.show()