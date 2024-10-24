import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers.legacy import Adam
from sklearn.metrics import root_mean_squared_error
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def main():

    #get and set number of epochs to train the model
    epochs = get_epochs()
    epochs = epochs
    
    # Load the data
    data=load_data()

    # Split the data into features and target
    X = data_train.iloc[:, :-1].values  # All columns except the last one
    y = data_train.iloc[:, 11].values   # 12th column (index 11)

    # Split the data into training+validation and testing sets
    #X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.0, random_state=42)
    X_train_val = X
    y_train_val = y
    X_test = data_test.iloc[:, :-1].values
    y_test = data_test.iloc[:, 11].values

    # Generate trial IDs
    num_datapoints = len(X_train_val)
    num_trials = num_datapoints // trial_size
    trial_ids = np.repeat(np.arange(num_trials), trial_size)

    # Shuffle the data while keeping trials intact
    X_train_val, y_train_val, trial_ids = shuffle(X_train_val, y_train_val, trial_ids, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    # Implement 5-fold cross-validation on the training+validation set
    gkf = GroupKFold(n_splits=5)#, shuffle=False)#, random_state=42)
    fold = 1
    val_losses = []
    rmses = []
    best_val_loss = np.inf
    best_model_weights = None

    for train_index, val_index in gkf.split(X_train_val, y_train_val, groups=trial_ids):
        print()
        print(f"===Fold {fold}===")
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]

        # Build the model
        model = Sequential()
        model.add(Dense(11, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dense(300, activation='relu', kernel_initializer='HeNormal'))
        model.add(Dropout(0.2))
        model.add(Dense(60, activation='relu', kernel_initializer='HeNormal'))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        # Compile the model
        optimizer = Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss='mean_absolute_error')
        
        # early stopping and reset the weights to the best model with the lowest validation loss
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=6, restore_best_weights=True) 
        #saves the best model of the fold based on the validation loss (including weights)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'best_model_{fold}.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

        # Train the model
        model.fit(X_train, y_train, epochs=epochs, batch_size=10, validation_data=(X_val, y_val), callbacks=[early_stopping, checkpoint])

        # Evaluate the model

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
    test_rmse = np.sqrt(root_mean_squared_error(y_test, y_test_pred))
    print(f'Test RMSE: {test_rmse}')

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

def get_epochs():
    while True:
        try:
            return int(input("Enter the number of epochs: "))
        except ValueError:
            pass

def load_data():
    import scipy.io

    # Load the .mat file
    mat = scipy.io.loadmat('/path/to/your/file.mat')

    # Print the keys of the dictionary
    print(mat.keys())

    # Assuming the data is stored in a variable named 'data' in the .mat file
    data = mat['data']

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv('/path/to/save/your/file.csv', index=False)

    # Return the DataFrame
    return df

if __name__ == "__main__":
    print("Test_BA_Data.py is being run directly")