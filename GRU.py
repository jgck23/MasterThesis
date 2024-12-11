import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, GRU, Input, Dropout, LSTM, Flatten, Bidirectional
from keras.optimizers.legacy import Adam
#from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import os
import wandb
from fun import load_data, set_optimizer, set_regularizer, data_leakage
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import shap

# padding wird auch  benötigt wenn alle Messungen gleich lang sind. Y_pred hat sehr kleine Werte für die ersten 10 Messungen. Lösung: padding und dann masking

os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()

def main():
    # Load the data
    fileName = 'NN_Bachelor_Thesis/ba_trials.csv'
    data = pd.read_csv(fileName, sep=',', header=None)

    # Split the data into features and target
    X = data.iloc[0:90000, 1:-1].values  # All features, the last 5 columns are not features
    y = data.iloc[0:90000, -1].values  # -5: wrist angle(x), -4: elbow angle(z), -3: shoulder flexion, -2: shoulder abduction, -1: Z-coordinate of right hand (height)
    trial_ids = data.iloc[0:90000, 0].values  # 1st column, trial IDs

    n_groups=trial_ids[-1] #get the last number of trial ids, which is the number of groups
    n_test_groups=round(0.2*n_groups)
    #print(n_test_groups)

    # Initialize GroupShuffleSplit and split the data
    gss = GroupShuffleSplit(n_splits=1, test_size=n_test_groups, random_state = 21)
    for train_index, test_index in gss.split(X, y, groups=trial_ids):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        trial_ids_train, trial_ids_test = trial_ids[train_index], trial_ids[test_index]

    # Check for trial leakage in train/test split
    data_leakage(trial_ids, train_index, test_index)

    # Standardize the data
    scaler_x = MinMaxScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    X_train_list = [X_train[i:i + 900] for i in range(0, len(X_train), 900)]
    X_test_list = [X_test[i:i + 900] for i in range(0, len(X_test), 900)]
    y_train_list = [y_train[i:i + 900] for i in range(0, len(y_train), 900)]
    y_test_list = [y_test[i:i + 900] for i in range(0, len(y_test), 900)]

    X_train, X_test, y_train, y_test = np.array(X_train_list), np.array(X_test_list), np.array(y_train_list), np.array(y_test_list)
    
    plt.hist(y_train.flatten(),bins=100)
    plt.show()
    plt.hist(y_test.flatten(),bins=100)
    plt.show()

    # Implement 5-fold cross-validation on the training+validation set
    kf = KFold(n_splits=3)
    fold = 1
    val_losses = []
    test_losses = []
    val_rmses = []
    test_rmses = []
    val_R2_scores = []
    test_R2_scores = []

    for i, (train_index, val_index) in enumerate(kf.split(X_train)):
        X_train_val, X_val = X_train[train_index], X_train[val_index]
        y_train_val, y_val = y_train[train_index], y_train[val_index]

        # Start a run, tracking hyperparameters
        wandb.init(
            project="BA_GRU",  # set the wandb project where this run will be logged
            group=os.environ["WANDB_RUN_GROUP"],  # group the runs together
            job_type="eval",  # job type
            # track hyperparameters and run metadata with wandb.config
            config={
                "Dataset": f'{fileName}',
                "layer_1": 128,
                #"activation_1": "relu",  # relu, sigmoid, tanh, softmax, softplus, softsign, selu, elu, exponential
                #"kernel_initializer_1": "HeNormal",  # HeNormal, GlorotNormal, LecunNormal, HeUniform, GlorotUniform, LecunUniform
                "layer_2": 64,
                #"kernel_initializer_2": "HeNormal",  # HeNormal, GlorotNormal, LecunNormal, HeUniform, GlorotUniform, LecunUniform
                #"activation_2": "relu",  # relu, sigmoid, tanh, softmax, softplus, softsign, selu, elu, exponential
                #"layer_3": 32,
                #"kernel_initializer_3": "HeNormal",  # HeNormal, GlorotNormal, LecunNormal, HeUniform, GlorotUniform, LecunUniform
                #"activation_3": "relu",  # relu, sigmoid, tanh, softmax, softplus, softsign, selu, elu, exponential
                "optimizer": "adam",  # adam, sgd, rmsprop, adagrad, adadelta, adamax, nadam, adamw
                "learning_rate": 0.001,
                "loss": "mean_absolute_error",
                "Dropout": 0.15,
                "epoch": 1000,
                "batch_size": 10,  # 20
                "FYI": "The saved model is the best model according to the lowest validation loss during training.",
            },
        )

        # [optional] use wandb.config as your config
        config = wandb.config

        model = Sequential()
        model.add(Input(shape=(900, X_train.shape[2])))
        model.add(GRU(config.layer_1, activation='relu', kernel_initializer='HeNormal', return_sequences=True, dropout=config.Dropout))#kernel_initializer=config.kernel_initializer_1,
        model.add(GRU(config.layer_2, activation='relu', kernel_initializer='HeNormal', return_sequences=True, dropout=config.Dropout))#kernel_initializer=config.kernel_initializer_2,
        model.add(Dense(1))  

        # Compile the model
        model.compile(set_optimizer(config.optimizer, config.learning_rate), loss=config.loss)

        # early stopping and reset the weights to the best model with the lowest validation loss after training
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=20,
            restore_best_weights=True,
        )
        # ModelCheckpoint callback to save the best model
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"best_model_{fold}.keras",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        )
        # Learning Rate Scheduler
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",  # Metric to monitor
            factor=0.5,  # Factor by which the learning rate will be reduced
            patience=6,  # Number of epochs with no improvement before reducing the learning rate
            min_lr=1e-6,  # Minimum learning rate (prevents reducing it too much)
            verbose=1  # Verbosity mode (1 = display updates)
        )

        # WandbMetricsLogger will log train and validation metrics to wandb
        # WandbModelCheckpoint will upload model checkpoints to wandb
        history = model.fit(
            x=X_train_val,
            y=y_train_val,
            epochs=config.epoch,
            batch_size=config.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                early_stopping,
                model_checkpoint,
                lr_scheduler,
                WandbMetricsLogger(log_freq="epoch"),
            ],
        )

        # load the model with the lowest validation loss and save it to wandb
        model.load_weights(f"best_model_{fold}.keras")
        wandb.save(f"best_model_{fold}.keras")

        # Evaluate the model on the validation set
        val_loss = model.evaluate(X_val, y_val)
        val_losses.append(val_loss)
        wandb.log({"Validation Loss": round(val_loss, 2)})

        # Predict the validation set results
        y_pred = model.predict(X_val)
        y_pred = np.squeeze(y_pred)

        # Calculate validation RMSE and validation R2 score, save them to wandb
        rmse = root_mean_squared_error(y_val.flatten(), y_pred.flatten())
        r2 = r2_score(y_val.flatten(), y_pred.flatten())
        val_rmses.append(rmse)
        val_R2_scores.append(r2)
        wandb.log({"Validation RMSE": round(rmse, 2)})
        wandb.log({"Validation R2 score": round(r2, 2)})

        # Evaluate the model on the test set and log loss
        test_loss = model.evaluate(X_test, y_test)
        test_losses.append(test_loss)
        wandb.log({'Test Loss': round(test_loss, 2)})

        # Calculate test RMSE and test R2 score, save them to wandb
        y_test_pred = model.predict(X_test)
        y_test_pred = np.squeeze(y_test_pred)
        test_rmse = root_mean_squared_error(y_test.flatten(), y_test_pred.flatten())
        test_r2 = r2_score(y_test.flatten(), y_test_pred.flatten())
        test_rmses.append(test_rmse)
        test_R2_scores.append(test_r2)
        wandb.log({'Test RMSE': round(test_rmse, 2)})
        wandb.log({'Test R2 score': round(test_r2, 2)})

        # Plot y_pred and y_test as a dot plot for the test set
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test.flatten(), y_test_pred.flatten(), alpha=0.5, label="Predicted vs Actual")
        plt.plot(
            [min(y_test.flatten()), max(y_test.flatten())],
            [min(y_test.flatten()), max(y_test.flatten())],
            color="red",
            label="Ideal line",
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        plt.legend()
        wandb.log({"Actual vs Predicted Values for test set": plt})
        plt.close()

        print(np.shape(y_test_pred))
        print(np.shape(y_test))

        # Plot a residual plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test.flatten(), y_test_pred.flatten() - y_test.flatten(), alpha=0.5, label="Residual Plot")
        plt.xlabel("Actual Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.legend()
        wandb.log({"Residual Plot": plt})
        plt.close()

        # print statements
        print(f"Fold {fold} - Validation Loss: {val_loss}")
        print(f"Fold {fold} - Test Loss: {test_loss}")
        print(f"Fold {fold} - Validation RMSE: {rmse}")
        print(f"Fold {fold} - Test RMSE: {test_rmse}")
        print(f"Fold {fold} - Validation R2 score: {r2}")
        print(f"Fold {fold} - Test R2 score: {test_r2}")

        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()

        fold += 1

    # Calculate average validation loss and RMSE
    avg_val_loss = np.mean(val_losses)
    avg_val_rmse = np.mean(val_rmses)
    avg_test_loss = np.mean(test_losses)
    avg_test_rmse = np.mean(test_rmses)
    avg_val_R2_score = np.mean(val_R2_scores)
    avg_test_R2_score = np.mean(test_R2_scores)
    print(f"Average Validation Loss: {avg_val_loss}")
    print(f"Average Test Loss: {avg_test_loss}")
    print(f"Average Validation RMSE: {avg_val_rmse}")
    print(f"Average Test RMSE: {avg_test_rmse}")
    print(f"Average Validation R2 score: {avg_val_R2_score}")
    print(f"Average Test R2 score: {avg_test_R2_score}")
    # best fold
    best_fold_loss = np.argmin(val_losses) + 1
    print(f"Best Fold according to validation loss: {best_fold_loss}")
    best_fold_rmse = np.argmin(val_rmses) + 1
    print(f"Best Fold according to validation RMSE: {best_fold_rmse}")

    # Log the aggregate metrics under the group
    wandb.init(project="BA_GRU", group=os.environ["WANDB_RUN_GROUP"], name="k_fold_summary")
    wandb.log({"avg_val_loss": avg_val_loss, "avg_test_loss": avg_test_loss, "avg_val_rmse": avg_val_rmse, "avg_test_rmse": avg_test_rmse, "avg_val_R2_score": avg_val_R2_score, "avg_test_R2_score": avg_test_R2_score, "best_fold_loss": best_fold_loss, "best_fold_rmse": best_fold_rmse})
    wandb.save("GRU.py")
    wandb.finish()
    
    model=tf.keras.models.load_model("best_model_2.keras")
    i=12 #max 19 for test size 20%

    sample=X_test[i,:,:]
    sample=sample.reshape(1,900,11)
    y_pred = model.predict(sample)
    y_pred = np.squeeze(y_pred)

    y_testw=y_test[i].flatten()
    y_pred=y_pred.flatten()
    y_testw=y_testw.reshape(-1,1)
    y_pred=y_pred.reshape(-1,1)
    y_testw=scaler_y.inverse_transform(y_testw)
    y_pred=scaler_y.inverse_transform(y_pred)

    plt.plot(y_testw, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.show()

    y_pred = model.predict(X_test)
    y_pred = np.squeeze(y_pred)
    y_pred = y_pred.flatten()
    y_test = y_test.flatten()
    y_pred = y_pred.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_pred)
    y_test = scaler_y.inverse_transform(y_test)
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
