#this file contains functions that are used in preprocessing and postprocessing steps during model training and evaluation
import scipy.io
import glob
import pandas as pd
import tensorflow as tf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import colorcet as cc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

#this sets the optimizer for the model with standard parameters if none are provided
def set_optimizer(optimizer, learning_rate=0.001, beta_1=0.9, beta_2=0.999):
    optimizer = optimizer.lower().strip()
    if optimizer == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2
        )
    elif optimizer == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.0, nesterov=False
        )
    elif optimizer == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
    elif optimizer == "adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer == "adadelta":
        return tf.keras.optimizers.Adadelta(learning_rate=learning_rate, rho=0.95)
    elif optimizer == "adamax":
        return tf.keras.optimizers.Adamax(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2
        )
    elif optimizer == "nadam":
        return tf.keras.optimizers.Nadam(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2
        )
    elif optimizer == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2
        )
    else:
        raise ValueError("Invalid optimizer")

#this sets the regularizer for the model with standard parameters if none are provided
def set_regularizer(regularizer, l1=0.01, l2=0.01):
    regularizer = regularizer.lower().strip()
    if regularizer == "l1":
        return tf.keras.regularizers.L1(l1)
    elif regularizer == "l2":
        return tf.keras.regularizers.L2(l2)
    elif regularizer == "l1_l2":
        return tf.keras.regularizers.L1L2(l1, l2)
    else:
        raise ValueError("Invalid regularizer")

#this sets the standardizer for the model 
def set_standardizer(standardizer): #add more if you want
    standardizer = standardizer.lower().strip()
    if standardizer == "standardscaler":
        return StandardScaler()
    elif standardizer == "minmaxscaler":
        return MinMaxScaler()
    elif standardizer == "robustscaler":
        return RobustScaler()
    elif standardizer == "quantiletransformer":
        return QuantileTransformer()
    else:
        raise ValueError("Invalid standardizer")

#this checks if the data is split correctly and if there is no data leakage between the training and test sets or between the training and validation sets
def data_leakage(trial_ids, train_index, test_index):
    train_trials = set(trial_ids[train_index])
    test_trials = set(trial_ids[test_index])
    common_trials = train_trials.intersection(test_trials)
    if common_trials:
        raise ValueError(
            f"Trial leakage detected between train and test sets for trials: {common_trials}"
        )

#this function plots the results of the models and compares them to the ground truth values
def plot_y(y_true, y_pred, trial_ids_test, target):
    if target == "WristAngle":
        zielvariable = "Handgelenkwinkel"
        target = "Wrist Angle"
    elif target == "ElbowAngle":
        zielvariable = "Ellenbogenwinkel"
        target = "Elbow Angle"
    elif target == "ShoulderAngleZ":
        zielvariable = "Schulterwinkel Z (Flexion/Extension)"
        target = "Shoulder Angle Z (Flexion/Extension)"
    elif target == "ShoulderAngleX":
        zielvariable = "Schulterwinkel X (Abduktion/Adduktion)"
        target = "Shoulder Angle X (Abduction/Adduction)"
    # Plot y_pred and y_test as a dot plot for the test set
    # Create a colormap for unique trial IDs
    unique_trials = np.unique(trial_ids_test)
    colors = cc.glasbey

    if len(unique_trials) > len(colors):
        raise ValueError(
            f"Number of unique trials {len(unique_trials)} exceeds the number of available colors {len(colors)}. Not all plots will be drawn. Adjust colormap accordingly."
        )

    ################## Plot y_pred and y_test as a line plot for the test set ###################
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
            name="Ideal Line",
        )
    )
    fig1.update_traces(opacity=0.75)
    fig1.update_layout(
        xaxis_title=f"Xsens {target} [°]", #Gemessene Werte {zielvariable}
        yaxis_title=f"Predicted {target} [°]", #Vorhergesagte Werte {zielvariable} 
        title=f"Dot Plot of Measured and Predicted Values of the {target}", #Plot der gemessenen und vorhergesagten Werte des {zielvariable} 
        title_font=dict(size=30),
        xaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
        yaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
    )

    ################# Plot residuals (y_pred - y_true) as a scatter plot ###################
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
            name="Ideal Line",
        )
    )
    fig2.update_traces(opacity=0.75)
    fig2.update_layout(
        xaxis_title=f"Xsens {target} [°]", #Gemessene Werte {zielvariable}
        yaxis_title="Residuals [°]",
        title="Residual Plot",
        title_font=dict(size=30),
        xaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
        yaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
    )

    ################### Plot y_true and y_pred as a line plot ###################
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
                name=f"Trial {trial_id} - Xsens Values",
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=np.arange(len(y_pred))[mask],
                y=y_pred[mask].flatten(),
                mode="lines",
                line=dict(color=color, dash="dash", width=1),
                name=f"Trial {trial_id} - Predicted Values",
            )
        )
    fig3.update_layout(
        xaxis_title="Data Points",
        yaxis_title=f"Xsens and Predicted {target}",
        title=f"Line Plot of Xsens and Predicted {target}", #Plot des gemessenen und vorhergesagten {zielvariable}
        title_font=dict(size=30),
        xaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
        yaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
    )

    return fig1, fig2, fig3

#this function plots a histogram of the target variable for the whole data, train and test.
def plot_y_hist(y, y_train, y_test):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=y, name="y"))
    fig.add_trace(go.Histogram(x=y_train, name="y_train"))
    fig.add_trace(go.Histogram(x=y_test, name="y_test"))
    fig.update_layout(
        xaxis_title="Werte (Values)",
        yaxis_title="Häufigkeit (Frequency)",
        title="Histogramm der Zielvariablen (Histogram of the target variable)",
    )
    # Overlay histograms
    fig.update_layout(barmode="overlay")
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.5)
    return fig

#this function plots a histogram of the height variable for the whole data, train and test.
def plot_height_hist(height, height_train, height_test):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=height, name="height"))
    fig.add_trace(go.Histogram(x=height_train, name="height_train"))
    fig.add_trace(go.Histogram(x=height_test, name="height_test"))
    fig.update_layout(
        xaxis_title="Height",
        yaxis_title="Frequency",
        title="Histogram of the height",
        title_font=dict(size=30),
        xaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
        yaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
    )
    # Overlay histograms
    fig.update_layout(barmode="overlay")
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.5)
    return fig

#this function plots the height vs the target variable
def plot_angle_vs_height(y, height, trial_ids, target):
    if target == "WristAngle":
        zielvariable = "Handgelenkwinkel"
        target = "Wrist Angle"
    elif target == "ElbowAngle":
        zielvariable = "Ellenbogenwinkel"
        target = "Elbow Angle"
    elif target == "ShoulderAngleZ":
        zielvariable = "Schulterwinkel Z (Flexion/Extension)"
        target = "Shoulder Angle Z (Flexion/Extension)"
    elif target == "ShoulderAngleX":
        zielvariable = "Schulterwinkel X (Abduktion/Adduktion)"
        target = "Shoulder Angle X (Abduction/Adduction)"

    fig = go.Figure()
    unique_trials = np.unique(trial_ids)
    colors = cc.glasbey

    for trial_id, color in zip(unique_trials, colors):
        mask = trial_ids == trial_id
        fig.add_trace(
            go.Scatter(
                x=y[mask].flatten(),
                y=height[mask].flatten(),
                mode="markers",
                marker=dict(color=color),
                name=f"Trial {trial_id}",
            )
        )
    fig.update_traces(opacity=0.75)
    fig.update_layout(
        xaxis_title=f"Xsens {target}", #Gemessener {zielvariable} (Xsens)
        yaxis_title="Drilling Height", #Bohrhöhe
        title=f"Plot of Height vs {target}", #Plot der Höhe gegen den {zielvariable}
        title_font=dict(size=30),
        xaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
        yaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
    )
    return fig

#this function plots the data before and after scaling
def plot_x_scaler(x, x_train, x_test, scaler):
    fig1 = go.Figure()
    for i in range(x.shape[1]):
        fig1.add_trace(go.Scatter(y=x[:, i], name=f"x_{i}", mode="lines"))
        fig1.update_layout(barmode="overlay")
    fig1.update_layout(
        xaxis_title="Datenpunkte (Data Points)",
        yaxis_title="Sensorwerte (Sensor values)",
        title="Plot der Daten (Plot of Data)",
    )

    fig2 = go.Figure()
    fig3 = go.Figure()
    for i in range(x_train.shape[1]):
        fig2.add_trace(go.Scatter(y=x_train[:, i], mode="lines"))
        fig3.add_trace(
            go.Scatter(y=scaler.inverse_transform(x_train)[:, i], mode="lines")
        )
        fig2.update_layout(barmode="overlay")
        fig3.update_layout(barmode="overlay")
    fig2.update_layout(
        xaxis_title="Datenpunkte (Data Points)",
        yaxis_title="Sensorwerte (Sensor values)",
        title="Plot der Trainingsdaten nach Skalierung (Plot of Training Data after Scaling)",
    )
    fig3.update_layout(
        xaxis_title="Datenpunkte (Data Points)",
        yaxis_title="Sensorwerte (Sensor values)",
        title="Plot der Trainingsdaten (Plot of Training Data)",
    )

    fig4 = go.Figure()
    fig5 = go.Figure()
    for i in range(x_test.shape[1]):
        fig4.add_trace(go.Scatter(y=x_test[:, i], mode="lines"))
        fig5.add_trace(
            go.Scatter(y=scaler.inverse_transform(x_test)[:, i], mode="lines")
        )
        fig4.update_layout(barmode="overlay")
        fig5.update_layout(barmode="overlay")
    fig4.update_layout(
        xaxis_title="Datenpunkte (Data Points)",
        yaxis_title="Sensorwerte (Sensor values)",
        title="Plot der Testdaten nach Skalierung (Plot of Test Data after scaling)",
    )
    fig5.update_layout(
        xaxis_title="Datenpunkte (Data Points)",
        yaxis_title="Sensorwerte (Sensor values)",
        title="Plot der Testdaten (Plot of Test Data)",
    )

    return fig1, fig2, fig3, fig4, fig5

#this function adds a trial identifier to the data, which can optionally be used as additional input to the model
def addtrialidentifier(X, trial_ids):
    # Convert trial IDs to integer indices
    unique_trials = np.unique(trial_ids)
    trial_to_index = {tid: i for i, tid in enumerate(unique_trials)}
    trial_indices = np.array([trial_to_index[tid] for tid in trial_ids])

    # Scale trial indices
    scaler = StandardScaler()
    trial_indices_scaled = scaler.fit_transform(trial_indices.reshape(-1, 1))

    # Add scaled trial indices to the features
    X_combined = np.hstack([trial_indices_scaled, X])

    return X_combined

#this function adds white noise to the data, which can be used to simulate sensor noise
def whitenoise(X, db):
    features = X[:, :-3] #sensels
    calculated_features = X[:, -3:] #sensor mean, total load, activated sensels

    signal_power = np.mean(features ** 2)
    noise_power = signal_power / (10 ** (db / 10))

    # Generate white noise with 0 mean and standard deviation noise_std
    noise = np.random.normal(loc=0, scale=np.sqrt(noise_power), size=features.shape)
    features_noisy = features + noise

    #recalculate the row means and total load
    row_means = np.round(features_noisy.mean(axis=1, keepdims=True), 2)
    total_load= np.sum(features_noisy, axis=1, keepdims=True)
    row_means.reshape(-1,1)
    total_load.reshape(-1,1)

    features_noisy = np.hstack((features_noisy, row_means))
    features_noisy = np.hstack((features_noisy, total_load))
    # add the actived sensels back to the noisy data
    noisy_data = np.hstack((features_noisy, calculated_features[:, 2].reshape(-1,1))) #assumes that sensels with a low activation are not activated even with noise

    '''fig = go.Figure()
    fig.add_trace(go.Scatter(y=features_noisy[:, 4], mode='lines', name='Noisy'))
    fig.update_traces(opacity=0.3)
    fig.add_trace(go.Scatter(y=features[:, 4], mode='lines', name='Original'))
    fig.show()'''

    return noisy_data