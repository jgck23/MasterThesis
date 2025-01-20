import Neural_Network as nn

path='Data/241212_Dataset_Leopard24_IMU.csv'

project_name='241212_Leopard24'

add_propsensor_features=False

height_filtering=False
height_lower=500
height_upper=1000

create_trial_identifier=False

variance_thresholding=True
variance_threshold=0.15

testdata_size=0.2
random_state=42 # any int in the range: [0, 2**32 - 1].

target='ElbowAngle' # WristAngle, ShoulderAngleZ (flexion/extension), ShoulderAngleX (abduction/adduction)

scaler_X='MinMaxScaler' # StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer

n_scross_val=5 # min 3

decrease_trials=False
decrease_trials_size= 0.2 # 0.2 uses only 20% of the original data, trials are randomly selected, pro Zeile 20% der Trials implementieren

decrease_duration=False
decrease_duration_size=0.7 # 0.7 uses only the initial 70% of the data of each trial

nn.main(path, height_filtering, height_lower, height_upper, create_trial_identifier, variance_thresholding, variance_threshold,
        testdata_size, target, scaler_X, add_propsensor_features, n_scross_val, decrease_trials, decrease_trials_size,
        decrease_duration, decrease_duration_size, project_name, random_state)