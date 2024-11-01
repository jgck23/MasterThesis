import numpy as np
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    GroupKFold,
)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers.legacy import Adam
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import os
import wandb
from fun import load_data, set_optimizer, set_regularizer
from wandb.integration.keras import (
    WandbMetricsLogger,
    WandbModelCheckpoint,
    WandbCallback,
)

os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()
def main():
    # Load the data
    data, name_mat = load_data()

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
    # y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    # y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # Implement 5-fold cross-validation on the training+validation set
    gkf = GroupKFold(n_splits=2)  # , shuffle=False)#, random_state=42)
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
            project="master-thesis-NN", # set the wandb project where this run will be logged
            group=os.environ["WANDB_RUN_GROUP"], # group the runs together
            job_type="eval", #job type
            # track hyperparameters and run metadata with wandb.config
            config={
                "name_mat": name_mat,
                "layer_1": 10,
                "activation_1": "selu", # relu, sigmoid, tanh, softmax, softplus, softsign, selu, elu, exponential
                "kernel_initializer_1": "HeNormal", # HeNormal, GlorotNormal, LecunNormal, HeUniform, GlorotUniform, LecunUniform
                "dropout": 0.2,  # random.uniform(0.01, 0.80),
                "layer_2": 100,
                "kernel_initializer_2": "HeNormal", # HeNormal, GlorotNormal, LecunNormal, HeUniform, GlorotUniform, LecunUniform
                "activation_2": "selu", # relu, sigmoid, tanh, softmax, softplus, softsign, selu, elu, exponential
                "layer_3": 100,
                "kernel_initializer_3": "HeNormal", # HeNormal, GlorotNormal, LecunNormal, HeUniform, GlorotUniform, LecunUniform
                "activation_3": "selu", # relu, sigmoid, tanh, softmax, softplus, softsign, selu, elu, exponential
                "optimizer": "adamw", # adam, sgd, rmsprop, adagrad, adadelta, adamax, nadam, adamw
                "loss": "mean_squared_error", # mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_logarithmic_error, cosine_similarity, huber, logcosh, poisson, kullback_leibler_divergence, hinge, squared_hinge, categorical_hinge, binary_crossentropy, kullback_leibler_divergence, poisson, cosine_proximity, is_categorical_crossentropy, sparse_categorical_crossentropy, binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, top_k_categorical_accuracy, sparse_top_k_categorical_accuracy, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_logarithmic_error, squared_hinge, hinge, categorical_hinge, logcosh, huber, cosine_similarity, cosine_proximity, poisson, kl_divergence, kullback_leibler_divergence, sparse_categorical_crossentropy, binary_crossentropy, is_categorical_crossentropy, sparse_categorical_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy, binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, top_k_categorical_accuracy, sparse_top_k_categorical_accuracy
                "epoch": 150,
                "batch_size": 20,
                "regularizer": "l1", # l1, l2, l1_l2
                "l1": 0.05, # lambda value for l1 regularization, lambda for l2 and l1_l2 can be set equally as well
                #"l2": 0.05,
                "FYI": "The saved model is the best model according to the lowest validation loss",

            },
        )

        # [optional] use wandb.config as your config
        config = wandb.config
        X_train_val, X_val = X_train[train_index], X_train[val_index]
        y_train_val, y_val = y_train[train_index], y_train[val_index]

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
        model.add(Dropout(config.dropout))
        model.add(
            Dense(
                config.layer_2,
                activation=config.activation_2,
                kernel_initializer=config.kernel_initializer_2,
                #kernel_regularizer=set_regularizer(config.regularizer, config.l1),
            )
        )
        model.add(Dropout(config.dropout))
        '''
        model.add(
            Dense(
                config.layer_3,
                activation=config.activation_3,
                kernel_initializer=config.kernel_initializer_3,
                #kernel_regularizer=set_regularizer(config.regularizer, config.l1),
            )
        )
        model.add(Dropout(config.dropout))#'''
        model.add(Dense(1))#, activation="relu"))

        # Compile the model
        model.compile(set_optimizer(config.optimizer), loss=config.loss)

        # early stopping and reset the weights to the best model with the lowest validation loss
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=10,
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

        # WandbMetricsLogger will log train and validation metrics to wandb
        # WandbModelCheckpoint will upload model checkpoints to wandb
        # save_name=f'best_model_{fold}.keras'
        history = model.fit(
            x=X_train_val,
            y=y_train_val,
            epochs=config.epoch,
            batch_size=config.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                early_stopping,
                model_checkpoint,
                WandbMetricsLogger(log_freq="epoch"),
            ],
        )

        # Evaluate the model on the validation set
        model.load_weights(f"best_model_{fold}.keras")
        val_loss = model.evaluate(X_val, y_val)
        val_losses.append(val_loss)
        wandb.log({"Validation Loss": round(val_loss, 2)})

        # Predict the validation set results
        y_pred = model.predict(X_val)

        # Calculate Validation RMSE
        rmse = root_mean_squared_error(y_val, y_pred)
        rmses.append(rmse)
        wandb.log({"Validation RMSE": round(rmse, 2)})

        # save the model
        wandb.save(f"best_model_{fold}.keras")
        
        # Evaluate the model on the test set and log loss
        test_loss = model.evaluate(X_test, y_test)
        y_test_pred = model.predict(X_test)
        wandb.log({'Test Loss': round(test_loss, 2)})

        # Calculate RMSE for the test set and log it
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        wandb.log({'Test RMSE': round(test_rmse, 2)})

        # Plot y_pred and y_test as a dot plot for the test set
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.5, label="Predicted vs Actual")
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
        wandb.log({"Actual vs Predicted Values for test set": plt})

        # print statements
        print(f"Fold {fold} - Validation Loss: {val_loss}")
        print(f"Fold {fold} - Test Loss: {test_loss}")
        print(f"Fold {fold} - Validation RMSE: {rmse}")
        print(f"Fold {fold} - Test RMSE: {test_rmse}")
        

        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()

        fold += 1

    # Calculate average validation loss and RMSE
    avg_val_loss = np.mean(val_losses)
    avg_rmse = np.mean(rmses)
    print(f"Average Validation Loss: {avg_val_loss}")
    print(f"Average RMSE: {avg_rmse}")
    # best fold
    best_fold_loss = np.argmin(val_losses) + 1
    print(f"Best Fold according to loss: {best_fold_loss}")
    best_fold_rmse = np.argmin(rmses) + 1
    print(f"Best Fold according to RMSE: {best_fold_rmse}")

if __name__ == "__main__":
    print("main.py is being run directly")
    main()
