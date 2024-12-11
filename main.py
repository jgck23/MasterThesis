import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    GroupKFold,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
import tensorflow as tf
# from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, GRU
from keras.optimizers.legacy import Adam
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import wandb
from fun import load_data, set_optimizer, set_regularizer, data_leakage, plot_y
from wandb.integration.keras import (
    WandbMetricsLogger,
    WandbModelCheckpoint,
    WandbCallback,
)
import shap

os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()
def main():
    # Load the data
    #fileName='Data/241113_Dataset_Leopard24.csv'
    fileName='NN_Bachelor_Thesis/ba_trials_extra.csv'
    data = pd.read_csv(fileName, sep=',', header=None)

    print(np.shape(data))

    height_preprocess = 0 # 1: upper trials only, -1: lower trials only, 0: all trials
    #data=data[data.iloc[:, -1] > 500] # watch out, this can lead to new all zero columns

    # Split the data into features and target
    X = data.iloc[:, 1:-2].values  # All features, the last 5 columns are not features
    y = data.iloc[:, -1].values  # -5: wrist angle(x), -4: elbow angle(z), -3: shoulder flexion, -2: shoulder abduction, -1: Z-coordinate of right hand (height)
    trial_ids = data.iloc[:, 0].values  # 1st column, trial IDs

    '''plt.hist(y, bins=100)
    plt.show()'''
    
    var_thres=False
    if var_thres: # deletes features with low variance, eg. lot of zeros and only a few non-zero values in one column
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        X=sel.fit_transform(X)

    n_groups=trial_ids[-1] #get the last number of trial ids, which is the number of groups
    n_test_groups=round(0.2*n_groups)
    #print(n_test_groups)

    # Initialize GroupShuffleSplit and split the data
    gss = GroupShuffleSplit(n_splits=1, test_size=n_test_groups, random_state = 21 )
    for train_index, test_index in gss.split(X, y, groups=trial_ids):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        trial_ids_train, trial_ids_test = trial_ids[train_index], trial_ids[test_index]

    # Check for trial leakage in train/test split
    data_leakage(trial_ids, train_index, test_index)

    # Standardize the data
    scaler_x = MinMaxScaler() #minmax when the features are on the same scale, standard scaler when they are not
    #target_scaler = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    #y_train = target_scaler.fit_transform(y_train.reshape(-1, 1))
    #y_test = target_scaler.transform(y_test.reshape(-1, 1))
    
    
    plt.hist(y_train, bins=100)
    plt.show()

    plt.hist(y_test, bins=100)
    plt.show()

    plt.plot(X_test[:,1:-5])
    plt.show()

    # Implement 5-fold cross-validation on the training+validation set
    gkf = GroupKFold(n_splits=3)
    fold = 1
    val_losses = []
    test_losses = []
    val_rmses = []
    test_rmses = []
    val_R2_scores=[]
    test_R2_scores=[]

    for train_index, val_index in gkf.split(X_train, y_train, groups=trial_ids_train):
        print()
        print(f"===Fold {fold}===")

        # Start a run, tracking hyperparameters
        wandb.init(
            project="BA_NN", # set the wandb project where this run will be logged
            group=os.environ["WANDB_RUN_GROUP"], # group the runs together
            job_type="eval", #job type
            # track hyperparameters and run metadata with wandb.config
            config={
                "Dataset": f'{fileName}',
                "layer_1": 32,
                "activation_1": "relu", # relu, sigmoid, tanh, softmax, softplus, softsign, selu, elu, exponential
                "kernel_initializer_1": "HeNormal", # HeNormal, GlorotNormal, LecunNormal, HeUniform, GlorotUniform, LecunUniform
                #"dropout": 0.2,  # random.uniform(0.01, 0.80),
                "layer_2": 64,
                "kernel_initializer_2": "HeNormal", # HeNormal, GlorotNormal, LecunNormal, HeUniform, GlorotUniform, LecunUniform
                "activation_2": "relu", # relu, sigmoid, tanh, softmax, softplus, softsign, selu, elu, exponential
                "layer_3": 32,
                "kernel_initializer_3": "HeNormal", # HeNormal, GlorotNormal, LecunNormal, HeUniform, GlorotUniform, LecunUniform
                "activation_3": "relu", # relu, sigmoid, tanh, softmax, softplus, softsign, selu, elu, exponential
                "optimizer": "adam", # adam, sgd, rmsprop, adagrad, adadelta, adamax, nadam, adamw
                "learning_rate": 0.005,
                "loss": "mean_squared_error", 
                "epoch": 200,
                "batch_size": 50, #20
                #"regularizer": "l1", # l1, l2, l1_l2
                #"l1": 0.01, # lambda value for l1 regularization, lambda for l2 and l1_l2 can be set equally as well
                #"l2": 0.01,
                "FYI": "The saved model is the best model according to the lowest validation loss during training.",
                "VarianceThreshold": var_thres,
                "height_preprocess": height_preprocess,
            },
        )

        # [optional] use wandb.config as your config
        config = wandb.config
        X_train_val, X_val = X_train[train_index], X_train[val_index]
        y_train_val, y_val = y_train[train_index], y_train[val_index]

        # Check for trial leakage in train/validation split for each fold
        data_leakage(trial_ids_train, train_index, val_index)

        # Build the model
        model = Sequential()
        model.add(Input(shape=(X_train_val.shape[1],)))
        model.add(
            Dense(
                config.layer_1,
                activation=config.activation_1,
                kernel_initializer=config.kernel_initializer_1,
                #kernel_regularizer=set_regularizer(config.regularizer, config.l1),
            )
        )
        #model.add(Dropout(config.dropout))
        model.add(
            Dense(
                config.layer_2,
                activation=config.activation_2,
                kernel_initializer=config.kernel_initializer_2,
                #kernel_regularizer=set_regularizer(config.regularizer, config.l1),
            )
        )
        #model.add(Dropout(config.dropout))
        #
        model.add(
            Dense(
                config.layer_3,
                activation=config.activation_3,
                kernel_initializer=config.kernel_initializer_3,
                #kernel_regularizer=set_regularizer(config.regularizer, config.l1),
            )
        )
        #model.add(Dropout(config.dropout))#
        model.add(Dense(1))#, activation = 'linear', kernel_initializer='GlorotUniform'))#, activation="relu"))

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
            factor=0.5,          # Factor by which the learning rate will be reduced
            patience=6,          # Number of epochs with no improvement before reducing the learning rate
            min_lr=1e-6,         # Minimum learning rate (prevents reducing it too much)
            verbose=1            # Verbosity mode (1 = display updates)
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
        #y_pred = target_scaler.inverse_transform(y_pred)

        # Calculate validation RMSE and validation R2 score, save them to wandb
        #y_val = target_scaler.inverse_transform(y_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        r2= r2_score(y_val, y_pred)
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
        #y_test_pred = target_scaler.inverse_transform(y_test_pred)
        #y_test = target_scaler.inverse_transform(y_test)
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        test_r2= r2_score(y_test, y_test_pred)
        test_rmses.append(test_rmse)
        test_R2_scores.append(test_r2)
        wandb.log({'Test RMSE': round(test_rmse, 2)})
        wandb.log({'Test R2 score': round(test_r2, 2)})

        plot1, plot2, plot3 = plot_y(y_test, y_test_pred, trial_ids_test)
        wandb.log({"Actual vs Predicted Values for test set": plot1})
        wandb.log({"Residual Plot": plot2})
        wandb.log({"Actual and Predicted Values line plot": plot3})

        '''# Create SHAP explainer
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
        # Compute mean absolute SHAP values per feature
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        # Save feature importance
        np.save("shap_feature_importance.npy", feature_importance)
        # Plot or print feature importance
        shap.summary_plot(shap_values, X_test)'''

        '''# Plot histogram of model weights per layer
        for layer in model.layers:
            if hasattr(layer, 'weights') and layer.get_weights():
                weights = layer.get_weights()[0]
                plt.figure(figsize=(10, 6))
                plt.hist(weights.flatten(), bins=50, alpha=0.75)
                plt.title(f'Layer {layer.name} Weights Distribution')
                plt.xlabel('Weight Value')
                plt.ylabel('Frequency')
                wandb.log({f'Layer {layer.name} Weights Distribution': plt})'''

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
    avg_val_R2_score=np.mean(val_R2_scores)
    avg_test_R2_score=np.mean(test_R2_scores)
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
    wandb.init(project="BA_NN", group=os.environ["WANDB_RUN_GROUP"], name="k_fold_summary")
    wandb.log({"avg_val_loss": avg_val_loss, "avg_test_loss": avg_test_loss, "avg_val_rmse": avg_val_rmse, "avg_test_rmse": avg_test_rmse, "avg_val_R2_score": avg_val_R2_score, "avg_test_R2_score": avg_test_R2_score, "best_fold_loss": best_fold_loss, "best_fold_rmse": best_fold_rmse})
    wandb.save("main.py")
    wandb.finish()

if __name__ == "__main__":
    main()