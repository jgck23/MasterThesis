import gpytorch
import torch
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import plotly.graph_objects as go
import numpy as np
import colorcet as cc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler
from sklearn.metrics import r2_score

# Define the model
class GPModel(gpytorch.models.ApproximateGP):  # Change ExactGP to ApproximateGP
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )  # Variational distribution
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )  # Variational strategy
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=5 / 2)
        )
        self.covar_module = self.base_covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
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

#def decrease_duration(data, trial_id, decrease_duration_size=0.4):
    trial_ids = trial_id
    unique_trials = np.unique(trial_ids)
    mask = []
    for trial in unique_trials:
        trial_indices = torch.nonzero(data[:, 0] == trial).squeeze()
        trial_length = len(trial_indices)
        mask.extend(trial_indices[: int(decrease_duration_size * trial_length)])
    data = data[mask,:]
    return data

fileName='Data/241212_Dataset_Leopard24_IMU.csv'
data = pd.read_csv(fileName, sep=',')

# Split the data into features and target
X = data.loc[:, "sensor1":"active_sensors"].values
y = data.loc[:, "ElbowAngle"].values
trial_ids = data.iloc[:, 0].values

# Initialize GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=21)

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

# Training data (convert to float32)
train_x = torch.tensor(X_train, dtype=torch.float64)
train_y = torch.tensor(y_train, dtype=torch.float64)
test_x = torch.tensor(X_test, dtype=torch.float64)
test_y = torch.tensor(y_test, dtype=torch.float64)

inducing_points = train_x[::250, :].clone()

# Model and likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
model = GPModel(inducing_points).double()

# Train with GPyTorch 
model.train()
likelihood.train()

# adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

# "Loss" for GPs
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

training_iter = 1000
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.base_covar_module.base_kernel.lengthscale.item(),
        likelihood.noise.item() 
    ))
    optimizer.step()

signal_variance = torch.var(train_y)
noise = likelihood.noise.item()
Signal_to_Noise = signal_variance / noise
print("Signal to Noise Ratio: ", Signal_to_Noise.item())

model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Make predictions
    observed_pred = likelihood(model(test_x))
    mean = observed_pred.mean
    lower, upper = observed_pred.confidence_region()

    # Calculate RMSE and R2
    rmse = torch.sqrt(torch.mean(torch.pow(mean - test_y, 2)))
    r2 = r2_score(test_y, mean)
    print("RMSE Test: ", rmse.item())
    print("R2 Test: ", r2)

    observed_pred_train = likelihood(model(train_x))
    mean_train = observed_pred_train.mean
    lower_train, upper_train = observed_pred_train.confidence_region()

    # Calculate RMSE and R2
    rmse_train = torch.sqrt(torch.mean(torch.pow(mean_train - train_y, 2)))
    r2_train = r2_score(train_y, mean_train)
    print("RMSE Train: ", rmse_train.item())
    print("R2 Train: ", r2_train)

plot1, plot2, plot3 = plot_y(y_test, mean, trial_ids_test)
plot1.show()
plot2.show()
plot3.show()


