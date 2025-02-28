from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filenamenn='Data/Post_Processing_Data/241212_Leopard_24_Split_NN_Elbow.csv'
filenamesgpr='Data/Post_Processing_Data/241212_Leopard_24_Split_SGPR_Elbow.csv'

datann = pd.read_csv(filenamenn, sep=',')
datann.columns =datann.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)
datasgpr = pd.read_csv(filenamesgpr, sep=',')
datasgpr.columns = datasgpr.columns.str.lower().str.replace(r'[\s_]', ' ', regex=True).str.replace(r'\bval\b', 'validation', regex=True)

vtb='test'
metric='rmse'

masknn = datann[f'{vtb} {metric}'].notna()
masksgpr = datasgpr[f'{vtb} {metric}'].notna()

y_sgpr = datasgpr.loc[masksgpr, f'{vtb} {metric}'].values
y_nn = datann.loc[masknn, f'{vtb} {metric}'].values

differenzen = y_nn - y_sgpr

sns.histplot(differenzen, kde=True)
plt.title("Histogramm der Differenzen mit KDE")
plt.xlabel("Differenz")
plt.ylabel("Häufigkeit")
plt.show()

stat, p_value = stats.shapiro(differenzen)
print("Shapiro-Wilk Statistik:", stat)
print("Shapiro-Wilk p-Wert:", p_value)

t_stat_rel, p_value_rel = stats.ttest_rel(y_nn, y_sgpr)
print("\nGepaarter t-Test")
print("T-Statistik:", t_stat_rel)
print("P-Wert:", p_value_rel)