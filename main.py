import Neural_Network as nn
import sgpr_torch as sgpr

path='Data/241212_Dataset_Leopard24_IMU.csv'

project_name='241212_Leopard24_TrialNum' # checking for the influence of the number of trials
#project_name='241212_Leopard24_Depth' #checking for the influence of the drilling depth relative for every trial
#project_name='241212_Leopard24_Split' # checking for the influence of the split of the data
#project_name='241212_Leopard24_wo_middle' # checking for the influence of the middle part of the data
#project_name='241212_241121_combined'
#project_name='241212_Leopard24_Optimisation'
project_name_sgpr = project_name + '_SGPR'

model_type='SGPR' 
#model_type='NN'

add_propsensor_features=False

height_filtering=False 
invert_selection=False # if True, the height values outside the range are selected
height_lower=400
height_upper=500

create_trial_identifier=False

variance_thresholding=True
variance_threshold=0.15

testdata_size=0.15
random_states=[88,99] # any int in the range: [0, 2**32 - 1]. Used for the GroupShuffleSplit. Change this to shuffle the data differently. [0,11,22,33,44,55,66,77,88,99]
seed=21 # any int in the range: [0, 2**32 - 1]. Seed for numpy when using decrease trials. Change this to shuffle the data differently.

target='ElbowAngle' # WristAngle, ShoulderAngleZ (flexion/extension), ShoulderAngleX (abduction/adduction)

scaler_X='MinMaxScaler' # StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer

n_scross_val=5 # min 3

decrease_trials=True
decrease_trials_sizes= [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] # 0.2 uses only 20% of the original data, trials are randomly selected

decrease_duration=False
decrease_duration_size= 1.0 # 0.7 uses only the initial 70% of the data of each trial
for random_state in random_states:
       for decrease_trials_size in decrease_trials_sizes:
                if model_type == 'NN':
                        nn.main(path, height_filtering, height_lower, height_upper, create_trial_identifier, variance_thresholding, variance_threshold,
                        testdata_size, target, scaler_X, add_propsensor_features, n_scross_val, decrease_trials, decrease_trials_size,
                        decrease_duration, decrease_duration_size, project_name, random_state, seed, invert_selection)
                elif model_type == 'SGPR':
                        sgpr.main(path, height_filtering, height_lower, height_upper, invert_selection, decrease_trials, decrease_trials_size,
                        decrease_duration, decrease_duration_size, project_name_sgpr, add_propsensor_features, target, create_trial_identifier, 
                        variance_thresholding, variance_threshold, testdata_size, random_state, scaler_X, n_scross_val, seed)
                else:
                        print('Model type not supported')