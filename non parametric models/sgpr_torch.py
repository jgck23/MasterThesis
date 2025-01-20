import gpytorch
import torch
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import plotly.graph_objects as go
import numpy as np
import colorcet as cc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, RobustScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

# Define the model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=5/2))

        inducing_points = train_x[::250,:].clone()
        self.inducing_points = torch.nn.Parameter(inducing_points)
        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, inducing_points=inducing_points, likelihood=likelihood)


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
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

X_test = torch.tensor(X_test, dtype=torch.float64)
y_test = torch.tensor(y_test, dtype=torch.float64)

# Early Stopping Configuration
early_stopping_patience = 10
best_val_loss = float('inf')
patience_counter = 0

gkf = GroupKFold(n_splits=2)
fold = 1
val_losses = []
test_losses = []
val_rmses = []
test_rmses = []
val_R2_scores = []
test_R2_scores = []
val_maes = []
test_maes = []

# Results storage
fold_results = []

for train_index, val_index in gkf.split(X_train, y_train, groups=trial_ids_train):
    print()
    print(f"===Fold {fold}===")
    X_train_val, X_val = X_train[train_index], X_train[val_index]
    y_train_val, y_val = y_train[train_index], y_train[val_index]

    X_train_val = torch.tensor(X_train_val, dtype=torch.float64)
    y_train_val = torch.tensor(y_train_val, dtype=torch.float64)
    X_val = torch.tensor(X_val, dtype=torch.float64)
    y_val = torch.tensor(y_val, dtype=torch.float64)

    # Reinitialize model and likelihood for each fold
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(X_train_val, y_train_val, likelihood)
    model.train()
    likelihood.train()
    
    # Optimizer and MLL
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Early Stopping Initialization
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, 1001):  # Max iterations set to 1000
        optimizer.zero_grad()
        output = model(X_train_val)
        loss = -mll(output, y_train_val)
        loss.backward()
        optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            val_output = model(X_val)
            val_loss = -mll(val_output, y_val).item()  # Validation loss
            
            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model state
                best_model_state = model.state_dict()
                best_likelihood_state = likelihood.state_dict()
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch} - Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Length Scale: {model.base_covar_module.base_kernel.lengthscale.item():.4f}, Noise: {likelihood.noise.item():.4f}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
        
        model.train()
        likelihood.train()

        # Load the best model for this fold
    model.load_state_dict(best_model_state)
    likelihood.load_state_dict(best_likelihood_state)
    
    # Evaluate on validation set
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        val_pred = likelihood(model(X_val)).mean
        test_pred = likelihood(model(X_test)).mean
        
        val_rmse = torch.sqrt(torch.mean((val_pred - y_val) ** 2)).item()
        val_r2 = r2_score(y_val.numpy(), val_pred.numpy())
        test_rmse = torch.sqrt(torch.mean((test_pred - y_test) ** 2)).item()
        test_r2 = r2_score(y_test.numpy(), test_pred.numpy())


        plot1, plot2, plot3 = plot_y(y_test, test_pred, trial_ids_test)
        plot1.show()
        plot2.show()
        plot3.show()

        signal_variance = torch.var(y_train_val)
        noise = model.likelihood.noise.item()
        Signal_to_Noise = signal_variance / noise
        print("Signal to Noise Ratio: ", Signal_to_Noise.item())
        
        # Store results for this fold
        fold_results.append({
            "fold": fold,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "test_rmse": test_rmse,
            "test_r2": test_r2
        })
        
        print(f"Fold {fold} - Val RMSE: {val_rmse:.4f}, Val R2: {val_r2:.4f}")
        print(f"Fold {fold} - Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}")
    
    fold += 1



