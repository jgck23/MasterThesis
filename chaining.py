import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import VarianceThreshold
from fun import *
import Neural_Network_ChainedVersion as nncv

fileName = 'Data/250318_Dataset_Eule3.csv'
testdata_size = 0.15
random_ints = [0,11,22,33,44,55,66,77,88,99] #important
scalewith = "MinMaxScaler"
add_prop_features = False
height_filtering = False
height_lower = 400
height_upper = 500
invert_selection = False
create_trial_identifier = False
var_thresholding = True
var_threshold = 0.15
n_cross_val = 5
decrease_trials = False
decrease_trials_size = 0.2
decrease_duration = False
decrease_duration_size = 0.7
project_name = "250318_Eule3_Chained"
seed = 21


for random_int in random_ints:
    ################## DATA HANDLING FOR CHAINING ##################
    data = pd.read_csv(fileName, sep=",")

    if height_filtering and invert_selection: # watch out, this can lead to new all zero columns
        selection = (data.loc[:, "RightHandZ"] > height_lower) & (data.loc[:, "RightHandZ"] < height_upper)
        data = data[~selection]
    elif height_filtering and not invert_selection: # watch out, this can lead to new all zero columns
        selection = (data.loc[:, "RightHandZ"] > height_lower) & (data.loc[:, "RightHandZ"] < height_upper)
        data = data[selection]

    if decrease_trials:
        trial_ids = data.loc[:, "Trial_ID"].values
        unique_trials = np.unique(trial_ids)
        n_trials = unique_trials.size
        n_trials = round(decrease_trials_size * n_trials)
        random_trials = np.random.choice(unique_trials, n_trials, replace=False) # set random seed !!!!!
        data = data[data["Trial_ID"].isin(random_trials)]

    if decrease_duration:
        trial_ids = data.loc[:, "Trial_ID"].values
        unique_trials = np.unique(trial_ids)
        mask = []
        for trial in unique_trials:
            trial_indices = data[data["Trial_ID"] == trial].index
            trial_length = len(trial_indices)
            mask.extend(trial_indices[: int(decrease_duration_size * trial_length)])
        data = data.loc[mask]

    total_datapoints=data.shape[0]
    print(f"Total number of datapoints: {total_datapoints}")

    X = data.loc[:, "sensor1":"active_sensors"].values
    if add_prop_features:
            X = pd.concat(
                [X, data.loc[:, "GunAX":"GunJZ"].values], axis=1
            )  # adds the xsens prop features to X data
    y = data.loc[:, "WristAngle":"ShoulderAngleZ"].values
    trial_ids = data.loc[:, "Trial_ID"].values

    if create_trial_identifier:
        X = addtrialidentifier(X, trial_ids)

    if var_thresholding:  # deletes features with low variance
        sel = VarianceThreshold(threshold=var_threshold)
        X = sel.fit_transform(X)

    n_groups = np.unique(
        trial_ids
    ).size  # get the number of unique trial ids, which is the number of groups
    n_test_groups = round(testdata_size * n_groups)

    # Initialize GroupShuffleSplit and split the data
    gss = GroupShuffleSplit(n_splits=1, test_size=n_test_groups, random_state=random_int)
    for train_index, test_index in gss.split(X, y, groups=trial_ids):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        trial_ids_train, trial_ids_test = trial_ids[train_index], trial_ids[test_index]

    # Check for trial leakage in train/test split
    data_leakage(trial_ids, train_index, test_index)

    ################## START OF THE CHAIN ##################
    scaler_x = set_standardizer(
        scalewith
    )  
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    predwrist=nncv.main(X_train_scaled, y_train[:, 0], trial_ids_train, X_test_scaled, y_test[:, 0], trial_ids_test, n_cross_val, decrease_trials, decrease_trials_size, decrease_duration, decrease_duration_size, project_name, random_int, seed, testdata_size, fileName, scalewith, n_groups, total_datapoints, create_trial_identifier, add_prop_features, var_thresholding, var_threshold, height_filtering, height_lower, height_upper, hidden_layers_num=3, hidden_layers_size=64, target='WristAngle', chaining=True)

    wristtrain = y_train[:, 0]
    X_train = np.column_stack((X_train, wristtrain))
    scaler_x = set_standardizer(
        scalewith
    )
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test = np.concatenate((X_test, predwrist), axis=1)
    X_test_scaled = scaler_x.transform(X_test)

    predelbow=nncv.main(X_train_scaled, y_train[:, 1], trial_ids_train, X_test_scaled, y_test[:, 1], trial_ids_test, n_cross_val, decrease_trials, decrease_trials_size, decrease_duration, decrease_duration_size, project_name, random_int, seed, testdata_size, fileName, scalewith, n_groups, total_datapoints, create_trial_identifier, add_prop_features, var_thresholding, var_threshold, height_filtering, height_lower, height_upper, hidden_layers_num=4, hidden_layers_size=128, target='ElbowAngle', chaining=True)
    elbowtrain = y_train[:, 1]
    X_train = np.column_stack((X_train, elbowtrain))
    scaler_x = set_standardizer(
        scalewith
    )
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test = np.concatenate((X_test, predelbow), axis=1)
    X_test_scaled = scaler_x.transform(X_test)

    predshoulder=nncv.main(X_train_scaled, y_train[:, 2], trial_ids_train, X_test_scaled, y_test[:, 2], trial_ids_test, n_cross_val, decrease_trials, decrease_trials_size, decrease_duration, decrease_duration_size, project_name, random_int, seed, testdata_size, fileName, scalewith, n_groups, total_datapoints, create_trial_identifier, add_prop_features, var_thresholding, var_threshold, height_filtering, height_lower, height_upper, hidden_layers_num=5, hidden_layers_size=128, target='ShoulderAngleZ', chaining=True)