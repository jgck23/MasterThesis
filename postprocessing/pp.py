from pfun import *
import pandas as pd


filename='Data/Post_Processing_Data/241212_Leopard24_Depth.csv'

data = pd.read_csv(filename, sep=',')
data.columns =data.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True)

#boxplot_error(data, 'Test R2 SCore', 'depth') #Test RMSE, Test MAE, Test R2 Score
plot_avg_values(data, 'avg test rmse', 'depth') #Test RMSE, Test MAE, Test R2 Score
#plot_NN(data, 'RMSE') #loss, R2, MAE, RMSE
