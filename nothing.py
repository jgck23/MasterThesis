import pandas as pd
import numpy as np

data = pd.read_csv('NN_Bachelor_Thesis/et_extra.csv', sep=';', header=None)

length = data.shape[0]
trials= length /900
print(length)
print(trials)

trial_num = np.zeros((180000, 1))
for i in range(int(trials)+1):
    trial_num[(i-1)*900:i*900] = int(i)

result = np.hstack((trial_num, data.values))
result = pd.DataFrame(result)

result.to_csv('NN_Bachelor_Thesis/ba_trials_extra.csv', index=False, header=False)
