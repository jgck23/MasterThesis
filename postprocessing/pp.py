from pfun import *
import pandas as pd
#!!! Dont forget to set the correct yaxis scale in the plot functions for single plots if you want to compare the models. 


#boxplot_error(data, 'Test R2 SCore', 'depth') #Test RMSE, Test MAE, Test R2 Score

##############TRIALNUM/DEPTH##############
#plot ntrials or depth variation plots
filenamenn='Data/Post_Processing_Data/241212_Leopard24_Depth_NN_Wrist.csv'
filenamesgpr='Data/Post_Processing_Data/241212_Leopard24_Depth_SGPR_Wrist.csv'
#filenamenn='Data/Post_Processing_Data/241212_Leopard24_TrialNum_NN_Elbow.csv'
#filenamesgpr='Data/Post_Processing_Data/241212_Leopard24_TrialNum_SGPR_Elbow.csv'

#data loading
datann = pd.read_csv(filenamenn, sep=',')
datann.columns =datann.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
datasgpr = pd.read_csv(filenamesgpr, sep=',')
datasgpr.columns = datasgpr.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
#flags:
metric = 'rmse' # RMSE, MAE, R2 score, loss
model = 'both' # NN, SGPR, both
plotfilepath = 'Data/Post_Processing_Data/plots'
mode = 'Depth' # Number of Holes, Depth
valtestboth = 'test' # validation, test, both (plots only the test or validation metric data or both, eg. compare the test and validation RMSE)
polydegree = 3 # polynomial degree for the fit

plot_ntrials_depth(datann, datasgpr, metric, valtestboth, model, mode, polydegree,plotfilepath)

##############SPLIT##############
#plot split section: always provide the data for the neural network and the sparse gaussian process regression for the same experiment. 
#experiment split with experiment split and depth with depth, etc.
filenamenn='Data/Post_Processing_Data/241212_Leopard24_Split_NN_Wrist.csv'
filenamesgpr='Data/Post_Processing_Data/241212_Leopard24_Split_SGPR_Wrist.csv'
#data loading
datann = pd.read_csv(filenamenn, sep=',')
datann.columns =datann.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
datasgpr = pd.read_csv(filenamesgpr, sep=',')
datasgpr.columns = datasgpr.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
#flags:
metric = 'rmse' # RMSE, MAE, R2 score, loss
model = 'both' # NN, SGPR, both
plotfilepath = 'Data/Post_Processing_Data/plots'
plotmeanabsolutedeviation = True # True, False (not possible for models='both' and valtestboth='both' since too many lines in one plot)
valtestboth = 'test' # validation, test, both (plots only the test or validation metric data or both, eg. compare the test and validation RMSE)

#plot_split(datann, datasgpr, metric, valtestboth, model, plotmeanabsolutedeviation, plotfilepath) #comment out if not needed

#################PLOT COMPARISON NN vs SGPR#################
filenamenn='Data/Post_Processing_Data/241212_Leopard24_Split_NN_Wrist.csv'
filenamesgpr='Data/Post_Processing_Data/241212_Leopard24_Split_SGPR_Wrist.csv'
#data loading
datann = pd.read_csv(filenamenn, sep=',')
datann.columns =datann.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
datasgpr = pd.read_csv(filenamesgpr, sep=',')
datasgpr.columns = datasgpr.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
#flags:
metric = 'rmse' # RMSE, MAE, R2 score, loss
vtb = 'test' # validation, test, both 
plotfilepath = 'Data/Post_Processing_Data/plots'
#plot_comparison_nnspgr(datann, datasgpr, metric, vtb, plotfilepath)