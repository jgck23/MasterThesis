#from fun import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process.kernels import ExpSineSquared

# Load the data
#data, name_mat = load_data()
#fileName='Data/241113_Dataset_Leopard24.csv'
fileName='NN_Bachelor_Thesis/ba_trials_extra.csv'
data = pd.read_csv(fileName, sep=',', header=None)

# Split the data into features and target
X = data.iloc[:, 1:-2].values 
y = data.iloc[:, -1].values  
trial_ids = data.iloc[:, 0].values 

# Initialize GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Split the data
for train_index, test_index in gss.split(X, y, groups=trial_ids):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    trial_ids_train = data.iloc[train_index, 0].values

# Standardize the data
scaler_x = StandardScaler()
#scaler_y = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

# Define the kernel
#kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
#kernel = RBF(length_scale=1.0)
# Define the exponential kernel
kernel = ExpSineSquared(length_scale=1.0, periodicity=3.0)

# Initialize GaussianProcessRegressor
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-2)

# Fit to the training data
gpr.fit(X_train, y_train)

# Predict on the test data
y_pred, sigma = gpr.predict(X_test, return_std=True)
testing_rmse = root_mean_squared_error(y_test, y_pred)
training_rmse=root_mean_squared_error(y_train, gpr.predict(X_train))
training_r2=r2_score(y_train, gpr.predict(X_train))
testing_r2=r2_score(y_test, y_pred)
print("training rmse:", training_rmse)
print("testing rmse:", testing_rmse)
print("training r2:", training_r2)
print("testing r2:", testing_r2)

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

plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()