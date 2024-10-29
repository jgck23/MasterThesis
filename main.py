import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers.legacy import Adam
from sklearn.metrics import root_mean_squared_error
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import random
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

def main():
    # Load the data
    data=load_data()
    
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
    #y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    #y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # Implement 5-fold cross-validation on the training+validation set
    gkf = GroupKFold(n_splits=5)#, shuffle=False)#, random_state=42)
    fold = 1
    val_losses = []
    rmses = []
    best_val_loss = np.inf
    best_model_weights = None

    for train_index, val_index in gkf.split(X_train, y_train, groups=trial_ids_train):
        print()
        print(f"===Fold {fold}===")

        # Start a run, tracking hyperparameters
        wandb.init(
            # set the wandb project where this run will be logged
            project="master-thesis-NN",

            # track hyperparameters and run metadata with wandb.config
            config={
                "layer_1": 1000,
                "activation_1": "relu",
                "kernel_initializer_1": "HeNormal",
                "dropout": 0.25, #random.uniform(0.01, 0.80),
                "layer_2": 1000,
                "kernel_initializer_2": "HeNormal",
                "activation_2": "relu",
                "layer_3": 500,
                "kernel_initializer_3": "HeNormal",
                "activation_3": "relu",
                "optimizer": "Adam",
                "loss": "mean_squared_error",
                #"metric": "accuracy",
                "epoch": 100,
                "batch_size": 30
            }
        )

        # [optional] use wandb.config as your config
        config = wandb.config
        X_train_val, X_val = X_train[train_index], X_train[val_index]
        y_train_val, y_val = y_train[train_index], y_train[val_index]

        # Build the model
        model = Sequential()
        model.add(Input(shape=(X_train_val.shape[1],)))
        model.add(Dense(config.layer_1, activation=config.activation_1, kernel_initializer=config.kernel_initializer_1))
        model.add(Dropout(config.dropout))
        model.add(Dense(config.layer_2, activation=config.activation_2, kernel_initializer=config.kernel_initializer_2))
        model.add(Dropout(config.dropout))
        model.add(Dense(config.layer_3, activation=config.activation_3, kernel_initializer=config.kernel_initializer_3))
        model.add(Dropout(config.dropout))
        model.add(Dense(1))

        # Compile the model
        optimizer = config.optimizer#(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss=config.loss)
        
        # early stopping and reset the weights to the best model with the lowest validation loss
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True) 
        #saves the best model of the fold based on the validation loss (including weights)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'best_model_{fold}.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

        # WandbMetricsLogger will log train and validation metrics to wandb
        # WandbModelCheckpoint will upload model checkpoints to wandb
        history = model.fit(x=X_train_val, y=y_train_val,
                            epochs=config.epoch,
                            batch_size=config.batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=[
                            early_stopping, 
                            checkpoint,
                            WandbMetricsLogger(log_freq="epoch"),
                            WandbModelCheckpoint("epoch-{epoch:02d}-val_loss-{val_loss:.2f}.keras", save_best_only=True)
                            ])

        # Evaluate the model on the validation set
        model.load_weights(f'best_model_{fold}.keras')
        val_loss = model.evaluate(X_val, y_val)
        val_losses.append(val_loss)
        print(f'Fold {fold} - Validation loss: {val_loss}')

        # Predict the validation set results
        y_pred = model.predict(X_val)

        # Calculate RMSE
        rmse = np.sqrt(root_mean_squared_error(y_val, y_pred))
        rmses.append(rmse)
        print(f'Fold {fold} - RMSE: {rmse}')

        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()

        fold += 1

    # Calculate average validation loss and RMSE
    avg_val_loss = np.mean(val_losses)
    avg_rmse = np.mean(rmses)
    print(f'Average Validation Loss: {avg_val_loss}')
    print(f'Average RMSE: {avg_rmse}')
    #best fold
    best_fold_loss = np.argmin(val_losses) + 1
    print(f'Best Fold according to loss: {best_fold_loss}')
    best_fold_rmse = np.argmin(rmses) + 1
    print(f'Best Fold according to RMSE: {best_fold_rmse}')

    # Load the best weights from cross-validation
    model.load_weights(f'best_model_{best_fold_loss}.keras')

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')
    y_test_pred = model.predict(X_test)
    
    # Calculate RMSE for the test set
    test_rmse = np.sqrt(root_mean_squared_error(y_test, y_test_pred))
    print(f'Test RMSE: {test_rmse}')

    '''
        f.write(f'Best Fold according to loss: {best_fold_loss}\n')
        f.write(f'Best Fold according to RMSE: {best_fold_rmse}\n')
        f.write(f'Test Loss: {test_loss}\n')
        f.write(f'Test RMSE: {test_rmse}\n')
    ''' 

    '''
    # Plot loss and val_loss over the epochs for the last fold
    plt.plot(model.history['loss'], label='Training Loss')
    plt.plot(model.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    '''

    # Plot y_pred and y_test as a dot plot for the test set
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5, label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Line')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.show()

def load_data():
    import scipy.io
    import glob
    while True:
        try:
            userinput= input("Enter the name of the file you want to create a model for : ")
            # Load the .mat file
            mat_file = glob.glob('**/'+str(userinput)+'.mat', recursive=True)
            if len(mat_file) > 1:
                raise ValueError("Multiple files found")

            mat = scipy.io.loadmat(mat_file[0]) 
            # Print the keys of the dictionary
            #print(mat.keys())
            # Assuming the data is stored in a variable named 'data' in the .mat file
            data = mat['learnerMatrix']
            # Convert the data to a pandas DataFrame
            df = pd.DataFrame(data)
            return df
        except FileNotFoundError:
            print("File not found")
            pass
        except ValueError:
            print("Invalid input")
            pass

if __name__ == "__main__":
    print("Test_BA_Data.py is being run directly")
    main()