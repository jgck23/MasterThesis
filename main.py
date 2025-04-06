import Neural_Network as nn
import sgpr_torch as sgpr

path='Data/250318_Dataset_Eule3.csv'

#project_name='241212_Leopard24_TrialNum' # checking for the influence of the number of trials
#project_name='241212_Leopard24_Depth' #checking for the influence of the drilling depth relative for every trial
#project_name='241212_Leopard24_Split' # checking for the influence of the split of the data
#project_name='241212_Leopard24_wo_middle' # checking for the influence of the middle part of the data
#project_name='241212_241121_combined'
#project_name='241212_Leopard24_WhiteNoise'
#project_name='241212_Leopard24_Optimisation'

#project_name='250312_Pferd12_Split'
#project_name='250312_Pferd12_TrialNum'
#project_name='250312_Pferd12_Depth'
#project_name='250312_Pferd12_WhiteNoise'

#project_name='250318_Eule3_Split'
project_name='250318_Eule3_TrialNum'
#project_name='250318_Eule3_Depth'
#project_name='250318_Eule3_Optimisation'
#project_name = '250318_Eule3_WhiteNoise' 

project_name_sgpr = project_name + '_SGPR'

model_type='SGPR' 
#model_type='NN'

target='WristAngle' # WristAngle, ElbowAngle, ShoulderAngleZ (flexion/extension), ShoulderAngleX (abduction/adduction)
hidden_layer_num=4 # only for NN, change the number of hidden layers
hidden_layer_size=128 #only for NN, change the number of neurons in the hidden layers
#3 @ 64 for wrist angle
#4 @ 128 for elbow angle
#5 @ 128 for shoulder angle z

learning_rate=0.01 # only for SGPR, change for NN in Neural_Network.py
#0.1 for wrist angle
#0.25 for elbow angle
#0.2 for shoulder angle z, could be higher maybe 0.225

add_propsensor_features=False

height_filtering=False 
invert_selection=False # if True, the height values outside the range are selected
height_lower=400
height_upper=500

create_trial_identifier=False

variance_thresholding=True
variance_threshold=0.15

testdata_size=0.15
random_states=[0,11,22,33,44,55,66,77,88,99] # any int in the range: [0, 2**32 - 1]. Used for the GroupShuffleSplit. Change this to shuffle the data differently. [0,11,22,33,44,55,66,77,88,99]
seed=21 # any int in the range: [0, 2**32 - 1]. Seed for numpy when using decrease trials. Change this to shuffle the data differently.

scaler_X='MinMaxScaler' # StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer

n_scross_val=5 # min 3

decrease_trials=True
decrease_trials_sizes= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # 0.2 uses only 20% of the original data, trials are randomly selected

decrease_duration=False
decrease_duration_sizes= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
 # 0.7 uses only the initial 70% of the data of each trial

add_white_noise=False
snr=30 #dB >30 almost no influence, 15-30 low noise, 5-15 high noise, <5 very high noise

for random_state in random_states:
        if decrease_duration and decrease_trials:
                raise ValueError('Both decrease_trials and decrease_duration are set to True. Please set only one to True.')
        elif decrease_trials:
                decrease_duration_size=1.0
                for decrease_trials_size in decrease_trials_sizes:
                        if model_type == 'NN':
                                nn.main(path, height_filtering, height_lower, height_upper, create_trial_identifier, variance_thresholding, variance_threshold,
                                testdata_size, target, scaler_X, add_propsensor_features, n_scross_val, decrease_trials, decrease_trials_size,
                                decrease_duration, decrease_duration_size, project_name, random_state, seed, invert_selection, hidden_layer_num, hidden_layer_size, snr, add_white_noise)
                        elif model_type == 'SGPR':
                                sgpr.main(path, height_filtering, height_lower, height_upper, invert_selection, decrease_trials, decrease_trials_size,
                                decrease_duration, decrease_duration_size, project_name_sgpr, add_propsensor_features, target, create_trial_identifier, 
                                variance_thresholding, variance_threshold, testdata_size, random_state, scaler_X, n_scross_val, seed, snr, add_white_noise, learning_rate)
                        else:
                                print('Model type not supported')
        elif decrease_duration:
                decrease_trials_size=1.0
                for decrease_duration_size in decrease_duration_sizes:
                        if model_type == 'NN':
                                nn.main(path, height_filtering, height_lower, height_upper, create_trial_identifier, variance_thresholding, variance_threshold,
                                testdata_size, target, scaler_X, add_propsensor_features, n_scross_val, decrease_trials, decrease_trials_size,
                                decrease_duration, decrease_duration_size, project_name, random_state, seed, invert_selection, hidden_layer_num, hidden_layer_size, snr, add_white_noise)
                        elif model_type == 'SGPR':
                                sgpr.main(path, height_filtering, height_lower, height_upper, invert_selection, decrease_trials, decrease_trials_size,
                                decrease_duration, decrease_duration_size, project_name_sgpr, add_propsensor_features, target, create_trial_identifier, 
                                variance_thresholding, variance_threshold, testdata_size, random_state, scaler_X, n_scross_val, seed, snr, add_white_noise, learning_rate)
                        else:
                                print('Model type not supported')
        else:
                decrease_duration_size=1.0
                decrease_trials_size=1.0
                if model_type == 'NN':
                        nn.main(path, height_filtering, height_lower, height_upper, create_trial_identifier, variance_thresholding, variance_threshold,
                        testdata_size, target, scaler_X, add_propsensor_features, n_scross_val, decrease_trials, decrease_trials_size,
                        decrease_duration, decrease_duration_size, project_name, random_state, seed, invert_selection, hidden_layer_num, hidden_layer_size, snr, add_white_noise)
                elif model_type == 'SGPR':
                        sgpr.main(path, height_filtering, height_lower, height_upper, invert_selection, decrease_trials, decrease_trials_size,
                        decrease_duration, decrease_duration_size, project_name_sgpr, add_propsensor_features, target, create_trial_identifier, 
                        variance_thresholding, variance_threshold, testdata_size, random_state, scaler_X, n_scross_val, seed, snr, add_white_noise, learning_rate)
                else:
                        print('Model type not supported')
        