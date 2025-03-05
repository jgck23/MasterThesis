import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from fun import *

data = pd.read_csv(fileName, sep=",")
X = data.loc[:, "sensor1":"active_sensors"].values
y = data.loc[:, "ElbowAngle":"ShoulderAngleZ"].values
trial_ids = data.loc[:, "Trial_ID"].values

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

scaler_x = set_standardizer(
    scalewith
)  # minmax when the features are on the same scale, standard scaler when they are not
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

# Check for trial leakage in train/test split
data_leakage(trial_ids, train_index, test_index)

bestmodelwrist, predwrist=nn.main(path, height_filtering, height_lower, height_upper, create_trial_identifier, variance_thresholding, variance_threshold, testdata_size, target, scaler_X, add_propsensor_features, n_scross_val, decrease_trials, decrease_trials_size, decrease_duration, decrease_duration_size, project_name, random_state, seed, invert_selection)

wristtrain = y_train[:, 0]
X_train = np.concatenate((X_train, wristtrain), axis=1)
scaler_x = set_standardizer(
    scalewith
)  # minmax when the features are on the same scale, standard scaler when they are not
X_train_scaled = scaler_x.fit_transform(X_train)
X_test = np.concatenate((X_test, predwrist), axis=1)
X_test_scaled = scaler_x.transform(X_test)

bestmodelelbow, predelbow=nn.main(path, height_filtering, height_lower, height_upper, create_trial_identifier, variance_thresholding, variance_threshold, testdata_size, target, scaler_X, add_propsensor_features, n_scross_val, decrease_trials, decrease_trials_size, decrease_duration, decrease_duration_size, project_name, random_state, seed, invert_selection)

elbowtrain = y_train[:, 1]
X_train = np.concatenate((X_train, elbowtrain), axis=1)
scaler_x = set_standardizer(
    scalewith
)  # minmax when the features are on the same scale, standard scaler when they are not
X_train_scaled = scaler_x.fit_transform(X_train)
X_test = np.concatenate((X_test, predelbow), axis=1)
X_test_scaled = scaler_x.transform(X_test)

bestmodelshoulder, predshoulder=nn.main(path, height_filtering, height_lower, height_upper, create_trial_identifier, variance_thresholding, variance_threshold, testdata_size, target, scaler_X, add_propsensor_features, n_scross_val, decrease_trials, decrease_trials_size, decrease_duration, decrease_duration_size, project_name, random_state, seed, invert_selection)
