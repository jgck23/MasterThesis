from pfun import *
import pandas as pd 

##############TRIALNUM/DEPTH##############
#filenamenn='Data/Post_Processing_Data/241212_Leopard24_Depth_NN.csv'
#filenamesgpr='Data/Post_Processing_Data/241212_Leopard24_Depth_SGPR.csv'
filenamenn='Data/Post_Processing_Data/241212_Leopard24_TrialNum_NN.csv'
filenamesgpr='Data/Post_Processing_Data/241212_Leopard24_TrialNum_SGPR.csv'

#filenamenn='Data/Post_Processing_Data/250312_Pferd12_Depth_NN.csv'
#filenamesgpr='Data/Post_Processing_Data/250312_Pferd12_Depth_SGPR.csv'
#filenamenn='Data/Post_Processing_Data/250312_Pferd12_TrialNum_NN.csv'
#filenamesgpr='Data/Post_Processing_Data/250312_Pferd12_TrialNum_SGPR.csv'

#filenamenn='Data/Post_Processing_Data/250318_Eule3_Depth_NN.csv'
#filenamesgpr='Data/Post_Processing_Data/250318_Eule3_Depth_SGPR.csv'
#filenamenn='Data/Post_Processing_Data/250318_Eule3_TrialNum_NN.csv'
#filenamesgpr='Data/Post_Processing_Data/250318_Eule3_TrialNum_SGPR.csv'

#data loading
datann = pd.read_csv(filenamenn, sep=',')
datann.columns =datann.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
datasgpr = pd.read_csv(filenamesgpr, sep=',')
datasgpr.columns = datasgpr.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
#flags:
metric = 'rmse' # RMSE, MAE, R2 score, loss
model = 'both' # NN, SGPR, both
plotfilepath = 'Data/Post_Processing_Data/plots'
mode = 'Number of Trials' # Number of Trials, Depth
valtestboth = 'test' # validation, test, both (plots only the test or validation metric data or both, eg. compare the test and validation RMSE)
polydegree = 3 # polynomial degree for the fit
target = 'ShoulderAngleZ' # WristAngle, ElbowAngle, ShoulderAngleZ, only one target
y_max = 40 # y axis max value, 20 for wrist and 40 for elbow and shoulder
avg = False # True: Plot only one average value for every x step, False: Plot all 50 values for every x step
offset = True # True: Offset markers for DNN and SGPR for no overlapping, False: No offset
plotdeptharea =False #Plot green background area for the minium required depth
depthareastart=30 #manually set the start of the area in %, e.g. 30% of the depth
plottrialnumimprovement = False #Plot two dashed lines for the improvement of the DNN
absoluteaxis = True #True: Absolute axis for the x axis, for Number of Trials: Number of Trials on the x axis, for Depth: Time in [s] on the x axis, False: Relative axis for the x axis, for Number of Trials: Number of Trials in % on the x axis, for Depth: Depth in % on the x axis

#plot_ntrials_depth(datann, datasgpr, metric, valtestboth, model, mode, polydegree,plotfilepath, target, y_max, avg, offset, plotdeptharea, depthareastart, plottrialnumimprovement, absoluteaxis) #comment out if not needed

#################PLOT COMPARISON NN vs SGPR#################
filenamenn='Data/Post_Processing_Data/250318_Eule3_Split_NN.csv'
filenamesgpr='Data/Post_Processing_Data/250318_Eule3_Split_SGPR.csv'
#data loading
datann = pd.read_csv(filenamenn, sep=',')
datann.columns =datann.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
datasgpr = pd.read_csv(filenamesgpr, sep=',')
datasgpr.columns = datasgpr.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
#flags:
metric = 'loss' # RMSE, MAE, R2 score, loss
vtb = 'test' # validation, test, both 
plotfilepath = 'Data/Post_Processing_Data/plots'
target = ['WristAngle','ElbowAngle','ShoulderAngleZ'] # WristAngle, ElbowAngle, ShoulderAngleZ, multiple targets possible
ymax=160 #20 for RMSE and MAE, 1 for R2 score
plot_comparison_nnspgr(datann, datasgpr, metric, vtb, plotfilepath, target, ymax) #comment out if not needed

#################Plot White Noise Comparison #################
fileWhiteNoiseNN='Data/Post_Processing_Data/241212_Leopard24_WhiteNoise_NN.csv'
fileWhiteNoiseSGPR='Data/Post_Processing_Data/241212_Leopard24_WhiteNoise_SGPR.csv'
fileNN='Data/Post_Processing_Data/241212_Leopard24_Split_NN.csv'
fileSGPR='Data/Post_Processing_Data/241212_Leopard24_Split_SGPR.csv'

#data loading
datawnnn = pd.read_csv(fileWhiteNoiseNN, sep=',')
datawnnn.columns =datawnnn.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
datawnsgpr = pd.read_csv(fileWhiteNoiseSGPR, sep=',')
datawnsgpr.columns = datawnsgpr.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
datann = pd.read_csv(fileNN, sep=',')
datann.columns =datann.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
datasgpr = pd.read_csv(fileSGPR, sep=',')
datasgpr.columns = datasgpr.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
#flags:
metric = 'rmse' # RMSE, MAE, R2 score, loss
vtb = 'test' # validation, test, both 
plotfilepath = 'Data/Post_Processing_Data/plots'
target = 'ElbowAngle' # only for Split Files to filter out the correct target, currently only the 'ElbowAngle' is used
#plot_white_noise(datawnnn, datawnsgpr,datann, datasgpr, metric, vtb, plotfilepath, target) #comment out if not needed






##############SPLIT############## !!!!!!!!!Deprecated!!!!!!!!!!!!! use the plot comparison nn vs sgpr function instead

#plot split section: always provide the data for the neural network and the sparse gaussian process regression for the same experiment. 
#experiment split with experiment split and depth with depth, etc.
filenamenn='Data/Post_Processing_Data/241212_Leopard24_Split_NN.csv'
filenamesgpr='Data/Post_Processing_Data/241212_Leopard24_Split_SGPR.csv'
#data loading
datann = pd.read_csv(filenamenn, sep=',')
datann.columns =datann.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
datasgpr = pd.read_csv(filenamesgpr, sep=',')
datasgpr.columns = datasgpr.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
#flags:
metric = 'loss' # RMSE, MAE, R2 score, loss
model = 'sgpr' # NN, SGPR, both
plotfilepath = 'Data/Post_Processing_Data/plots'
plotmeanabsolutedeviation = False # True, False (not possible for models='both' and valtestboth='both' since too many lines in one plot)
valtestboth = 'both' # validation, test, both (plots only the test or validation metric data or both, eg. compare the test and validation RMSE)
target='ElbowAngle' # WristAngle, ElbowAngle, ShoulderAngleZ, only one target possible

#plot_split(datann, datasgpr, metric, valtestboth, model, plotmeanabsolutedeviation, plotfilepath, target) #comment out if not needed