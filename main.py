# this is the main file to run the experiments for the DNN and the SGPR (train-test split, Number of Trials, Depth, White Noise)
import Neural_Network as nn
import sgpr_torch as sgpr

path='Data/250318_Dataset_Eule3.csv' #specify the path to the dataset

############### EXPERIMENT NAME FOR WANDB LOGGING ################
#project_name='241212_Leopard24_TrialNum' # checking for the influence of the number of trials
#project_name='241212_Leopard24_Depth' #checking for the influence of the drilling depth relative for every trial
#project_name='241212_Leopard24_Split' # checking for the influence of the split of the data
#project_name='241212_Leopard24_WhiteNoise'

#project_name='250312_Pferd12_TrialNum'
#project_name='250312_Pferd12_Depth'
#project_name='250312_Pferd12_Split'
#project_name='250312_Pferd12_WhiteNoise'

project_name='250318_Eule3_TrialNum'
#project_name='250318_Eule3_Depth'
#project_name='250318_Eule3_Split'
#project_name = '250318_Eule3_WhiteNoise' 

project_name_sgpr = project_name + '_SGPR' #for the SGPR the project name is automatically set here!

################ MODEL TYPE AND TARGET SELECTION ################
model_type='SGPR' 
#model_type='NN'

target='WristAngle' # WristAngle, ElbowAngle, ShoulderAngleZ (flexion/extension), ShoulderAngleX (abduction/adduction)
hidden_layer_num=4 # only for NN, change the number of hidden layers
hidden_layer_size=128 #only for NN, change the number of neurons in the hidden layers
#3 @ 64 for wrist angle
#4 @ 128 for elbow angle
#5 @ 128 for shoulder angle z

# the settings of SGPR for the different participants and angles can be found in the .cvs-files from WandB and in the python files that are saved in the cross_valsummaries (Messdaten Server IPEK)
learning_rate=0.01 # only for SGPR, change for NN in Neural_Network.py
early_stopping_patience=35 # only for SGPR, change for NN in Neural_Network.py !!!line 273 in sgpr_torch.py: mechanism to adjust this value dynamically when more than 50 epochs are trained!!! (to reduce training time)
early_stopping_minimum_delta = 2e-2 # only for SGPR, change for NN in Neural_Network.py
max_epochs= 100 # only for SGPR, change for NN in Neural_Network.py
learning_rate_scheduler_patience = 12 # only for SGPR, change for NN in Neural_Network.py, after the patience the learning rate is halved

################# ADDITIONAL PARAMETERS AND PREPROCESSING STEPS ################
add_propsensor_features=False # optionally add the features from the propsensor of the hammer drill
height_filtering=False # if True, the data is filtered by the relative height of the Xsens right hand height
invert_selection=False # if True, the height values outside the range are selected
height_lower=400
height_upper=500
create_trial_identifier=False # create a trial identifier that can optionally be used as additional feature
variance_thresholding=True # if True, the variance thresholding is applied to the data
variance_threshold=0.15 # set the variance threshold
testdata_size=0.15 # set the size of the test data in the range [0,1]
seed=21 # any int in the range: [0, 2**32 - 1]. Seed for numpy when using decrease trials. Change this to shuffle the data differently. But leave it at 21.
scaler_X='MinMaxScaler' # StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
n_scross_val=5 # set the number of folds for the cross validation, min. 3

################ EXPERIMENTAL SETTINGS ################
random_states=[0,11,22,33,44,55,66,77,88,99]# any int in the range: [0, 2**32 - 1]. Used for the GroupShuffleSplit. Change this to shuffle the data differently. [0,11,22,33,44,55,66,77,88,99]

decrease_trials=False
decrease_trials_sizes= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 

decrease_duration=False
decrease_duration_sizes= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

add_white_noise=False
snr=30 #dB >30 almost no influence, 15-30 low noise, 5-15 high noise, <5 very high noise

################ RUN THE EXPERIMENTS ################ (leave this as it is)
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
                                variance_thresholding, variance_threshold, testdata_size, random_state, scaler_X, n_scross_val, seed, snr, add_white_noise, learning_rate, early_stopping_patience, early_stopping_minimum_delta, max_epochs, learning_rate_scheduler_patience)
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
                                variance_thresholding, variance_threshold, testdata_size, random_state, scaler_X, n_scross_val, seed, snr, add_white_noise, learning_rate, early_stopping_patience, early_stopping_minimum_delta, max_epochs, learning_rate_scheduler_patience)
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
                        variance_thresholding, variance_threshold, testdata_size, random_state, scaler_X, n_scross_val, seed, snr, add_white_noise, learning_rate, early_stopping_patience, early_stopping_minimum_delta, max_epochs, learning_rate_scheduler_patience)
                else:
                        print('Model type not supported')
        