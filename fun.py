import scipy.io
import glob
import pandas as pd
import tensorflow as tf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import colorcet as cc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer


def load_data(userinput=None):
    while True:
        try:
            if userinput is None:
                userinput = input(
                    "Enter the name of the file you want to create a model for : "
                )
            # Load the .mat file
            mat_file = glob.glob("**/" + str(userinput) + ".mat", recursive=True)
            if len(mat_file) > 1:
                raise ValueError("Multiple files found")

            mat = scipy.io.loadmat(mat_file[0])
            # Print the keys of the dictionary
            # print(mat.keys())
            # Assuming the data is stored in a variable named 'data' in the .mat file
            data = mat["learnerMatrix"]
            # Convert the data to a pandas DataFrame
            df = pd.DataFrame(data)
            return df, userinput
        except FileNotFoundError:
            print("File not found")
            pass
        except ValueError:
            print("Invalid input")
            pass


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


def data_leakage(trial_ids, train_index, test_index):
    train_trials = set(trial_ids[train_index])
    test_trials = set(trial_ids[test_index])
    common_trials = train_trials.intersection(test_trials)
    if common_trials:
        raise ValueError(
            f"Trial leakage detected between train and test sets for trials: {common_trials}"
        )


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

def whitenoise(X, noise_std):
    features = X[:, :-3] #sensels
    calculated_features = X[:, -3:] #sensor mean, total load, activated sensels

    # Generate white noise with 0 mean and standard deviation noise_std
    noise = np.random.normal(loc=0, scale=noise_std, size=features.shape)
    features_noisy = features + noise

    #recalculate the row means and total load
    row_means = np.round(features_noisy.mean(axis=1, keepdims=True), 2)
    total_load= np.sum(features_noisy, axis=1, keepdims=True)

    features_noisy = np.hstack((features_noisy, row_means))
    features_noisy = np.hstack((features_noisy, total_load))
    # add the actived sensels back to the noisy data
    noisy_data = np.hstack((features_noisy, calculated_features[:, 2]))

    return noisy_data