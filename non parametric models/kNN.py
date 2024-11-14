import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, root_mean_squared_error 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from fun import load_data

# Load the data
data, name_mat = load_data()
#data= pd.read_csv('Data/EHKL.csv', sep=',')

# Split the data into features and target
X = data.iloc[:, 1:101].values  # All features, columns 1 to 100
y = data.iloc[:, 101].values  # 101th column, elbow flexion angle
trial_ids = data.iloc[:, 0].values  # 1st column, trial IDs

# Initialize GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Split the data
for train_index, test_index in gss.split(X, y, groups=trial_ids):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    trial_ids_train = data.iloc[train_index, 0].values

# Standardize the data
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

train_rmse=[]
test_rmse=[]
k_neighbors=750
# Create and train KNN 
for i in range (1,k_neighbors):
    knn_regressor = KNeighborsRegressor(n_neighbors=i)
    knn_regressor.fit(X_train, y_train)
    y_pred = knn_regressor.predict(X_test)
    # Evaluate the model
    testing_rmse = root_mean_squared_error(y_test, y_pred)
    training_rmse=root_mean_squared_error(y_train, knn_regressor.predict(X_train))
    print("training rmse for k="+str(i)+":", training_rmse) 
    print("testing rmse for k="+str(i)+":", testing_rmse)
    print(" ")
    #print(testing_rmse)
    #print(training_rmse)

    test_rmse.append(testing_rmse)
    train_rmse.append(training_rmse)

# Plot accuracy, optimal k is ca. 600
plt.plot(range(1, k_neighbors), test_rmse, label='Test RMSE')
plt.plot(range(1, k_neighbors), train_rmse, label='Train RMSE')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('RMSE')
plt.title('RMSE vs Number of Neighbors')
plt.legend()
plt.show()

# get index of smallest test rmse
min_rmse_index = np.argmin(test_rmse)
print("Optimal k: ", min_rmse_index+1)
# create knn regessor with optimal k
knn_regressor = KNeighborsRegressor(n_neighbors=min_rmse_index+1)
knn_regressor.fit(X_train, y_train)
y_pred = knn_regressor.predict(X_test)

# Plot y_pred and y_test as a dot plot for the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted vs Actual")
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color="red",
    label="Ideal Line",
)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()