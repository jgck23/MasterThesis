# this is the chained version of Neural_Network.py
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GroupKFold,
)
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import os
import wandb
from fun import *
from wandb.integration.keras import WandbMetricsLogger
import random

os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()


def main(
    X_train,
    y_train,
    trial_ids_train,
    X_test,
    y_test,
    trial_ids_test,
    n_cross_val,
    decrease_trials,
    decrease_trials_size,
    decrease_duration,
    decrease_duration_size,
    project_name,
    random_int,
    npseed,
    testdata_size,
    fileName,
    scalewith,
    n_groups,
    total_datapoints,
    create_trial_identifier,
    add_prop_features,
    var_thresholding,
    var_threshold,
    height_filtering,
    height_lower,
    height_upper,
    hidden_layers_num,
    hidden_layers_size,
    target,
    chaining=False,
):
    np.random.seed(npseed)
    tf.random.set_seed(42)
    random.seed(42)

    # Implement 5-fold cross-validation on the training+validation set
    gkf = GroupKFold(n_splits=n_cross_val)
    fold = 1
    val_losses = []
    test_losses = []
    val_rmses = []
    test_rmses = []
    val_R2_scores = []
    test_R2_scores = []
    val_maes = []
    test_maes = []
    predicted_values = []

    for train_index, val_index in gkf.split(X_train, y_train, groups=trial_ids_train):
        print()
        print(f"===Fold {fold}===")

        # Start a run, tracking hyperparameters
        wandb.init(
            project=project_name,  # set the wandb project where this run will be logged
            group=os.environ["WANDB_RUN_GROUP"],  # group the runs together
            job_type="eval",  # job type
            settings=wandb.Settings(silent=True),
            # track hyperparameters and run metadata with wandb.config
            config={
                "Dataset": fileName,
                "target": target,
                "scaler": scalewith,
                "n_splits_cross_val": n_cross_val,
                "test_size": testdata_size,
                "trial_number": n_groups,
                "total_datapoints": total_datapoints,
                "create_trial_identifier": create_trial_identifier,
                "add_prop_features": add_prop_features,
                "random_state": random_int,
                "np_seed": npseed,
                "decrease_trials": decrease_trials,
                "decrease_trials_size": decrease_trials_size,
                "decrease_duration": decrease_duration,
                "decrease_duration_size": decrease_duration_size,
                "activation": "relu",  # relu, sigmoid, tanh, softmax, softplus, softsign, selu, elu, exponential
                "kernel_initializer": "HeNormal",  # HeNormal, GlorotNormal, LecunNormal, HeUniform, GlorotUniform, LecunUniform
                "dropout": 0.15,
                "hidden_layers_num": hidden_layers_num,
                "hidden_layers_size": hidden_layers_size,
                "optimizer": "adam",  # adam, sgd, rmsprop, adagrad, adadelta, adamax, nadam, adamw
                "learning_rate": 0.0005,
                "loss": "mean_squared_error",
                "epoch": 1000,
                "batch_size": 64,  # 20
                "regularizer_type": "l1",  # l1, l2, l1_l2
                "l": 0.001,  # lambda value for l1 regularization, lambda for l2 and l1_l2 can be set equally as well
                "FYI": "The saved model is the best model according to the lowest validation loss during training.",
                "VarianceThreshold": var_thresholding,
                "variance_threshold": var_threshold,
                "height_preprocess": height_filtering,
                "height_lower": height_lower,
                "height_upper": height_upper,
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

        for i in range(config.hidden_layers_num):
            model.add(
                Dense(
                    config.hidden_layers_size,
                    activation=config.activation,
                    kernel_initializer=config.kernel_initializer,
                    kernel_regularizer=set_regularizer(
                        config.regularizer_type, config.l
                    ),
                )
            )
            model.add(Dropout(config.dropout))

        model.add(
            Dense(1)
        )  # , activation = 'linear', kernel_initializer='GlorotUniform'))#, activation="relu"))

        # Compile the model
        model.compile(
            set_optimizer(config.optimizer, config.learning_rate), loss=config.loss
        )

        # early stopping and reset the weights to the best model with the lowest validation loss after training
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=25,
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
            verbose=1,  # Verbosity mode (1 = display updates)
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
        # y_pred = target_scaler.inverse_transform(y_pred)

        # Calculate validation RMSE and validation R2 score, save them to wandb
        # y_val = target_scaler.inverse_transform(y_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        val_rmses.append(rmse)
        val_R2_scores.append(r2)
        val_maes.append(mae)
        wandb.log({"Validation RMSE": round(rmse, 2)})
        wandb.log({"Validation R2 score": round(r2, 2)})
        wandb.log({"Validation MAE": round(mae, 2)})

        # Evaluate the model on the test set and log loss
        test_loss = model.evaluate(X_test, y_test)
        test_losses.append(test_loss)
        wandb.log({"Test Loss": round(test_loss, 2)})

        # Calculate test RMSE and test R2 score, save them to wandb
        y_test_pred = model.predict(X_test)
        # y_test_pred = target_scaler.inverse_transform(y_test_pred)
        # y_test = target_scaler.inverse_transform(y_test)
        predicted_values.append(y_test_pred)
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmses.append(test_rmse)
        test_R2_scores.append(test_r2)
        test_maes.append(test_mae)
        wandb.log({"Test RMSE": round(test_rmse, 2)})
        wandb.log({"Test R2 score": round(test_r2, 2)})
        wandb.log({"Test MAE": round(test_mae, 2)})

        plot1, plot2, plot3 = plot_y(y_test, y_test_pred, trial_ids_test, target)
        wandb.log({"Actual vs Predicted Values for test set": plot1})
        wandb.log({"Residual Plot": plot2})
        wandb.log({"Actual and Predicted Values line plot": plot3})

        # print statements
        print(f"{'Metric':<20} {'Validation':<15} {'Test':<15}")
        print(f"{'-'*50}")
        print(f"{'Loss':<20} {val_loss:<15.4f} {test_loss:<15.4f}")
        print(f"{'RMSE':<20} {rmse:<15.4f} {test_rmse:<15.4f}")
        print(f"{'R2 score':<20} {r2:<15.4f} {test_r2:<15.4f}")
        print(f"{'MAE':<20} {mae:<15.4f} {test_mae:<15.4f}")

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
    avg_val_mae = np.mean(val_maes)
    avg_test_mae = np.mean(test_maes)
    # Print the average metrics in a table format
    print(f"{'Metric':<25} {'Validation':<15} {'Test':<15}")
    print(f"{'-'*55}")
    print(f"{'Average Loss':<25} {avg_val_loss:<15.4f} {avg_test_loss:<15.4f}")
    print(f"{'Average RMSE':<25} {avg_val_rmse:<15.4f} {avg_test_rmse:<15.4f}")
    print(
        f"{'Average R2 score':<25} {avg_val_R2_score:<15.4f} {avg_test_R2_score:<15.4f}"
    )
    print(f"{'Average MAE':<25} {avg_val_mae:<15.4f} {avg_test_mae:<15.4f}")
    # best fold
    best_fold_loss = np.argmin(val_losses) + 1
    best_fold_rmse = np.argmin(val_rmses) + 1
    print(f"Best Fold according to validation loss: {best_fold_loss}")
    print(f"Best Fold according to validation RMSE: {best_fold_rmse}")

    # Logging the aggregate metrics under the same group as the cross-validation runs
    wandb.init(
        project=project_name,
        group=os.environ["WANDB_RUN_GROUP"],
        name="crossval_summary",
        settings=wandb.Settings(silent=True),
    )
    wandb.log(
        {
            "avg_val_loss": avg_val_loss,
            "avg_test_loss": avg_test_loss,
            "avg_val_rmse": avg_val_rmse,
            "avg_test_rmse": avg_test_rmse,
            "avg_val_R2_score": avg_val_R2_score,
            "avg_test_R2_score": avg_test_R2_score,
            "avg_val_mae": avg_val_mae,
            "avg_test_mae": avg_test_mae,
            "best_fold_loss": best_fold_loss,
            "best_fold_rmse": best_fold_rmse,
        }
    )
    wandb.save("Neural_Network.py")  # save the script
    wandb.finish()

    if chaining:
        return predicted_values[np.argmin(test_rmses)]


if __name__ == "__main__":
    main()
