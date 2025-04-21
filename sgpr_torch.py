# sgpr logic after: https://docs.gpytorch.ai/en/v1.12/examples/02_Scalable_Exact_GPs/SGPR_Regression_CUDA.html (last accessed 2025-04-14)
# this is the file for the SGPR model with wandb logging
import gpytorch
import torch
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from fun import *
from sklearn.feature_selection import VarianceThreshold
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import os

# Inducing point logic, selecting every 500th point seems like a lot for 65300 total points. But when leaving the test set
# and validation set out, respectively 20% and for 2 fold cross validation 50%, that leaves roughly 26.200 points for
# training. Selecting every 500th poinnt then results in 52 inducing points, which can be not enough.

# https://github.com/cornellius-gp/gpytorch/issues/1787 saving of the hyperparameters is possible but not the posterior,
# so the training data has to be instantiated with the hyperparams to make predictions, still doesnt work
# https://github.com/cornellius-gp/gpytorch/issues/1308 all possible solutions have been tested but the val loss after reloading
# the model with the lowest val loss while training is always different than the lowest val loss during training. Not
# significant but still a problem.


# Define the model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        self.base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=5 / 2)  # define the kernel
        )

        inducing_points = train_x[
            ::75,
            :,  # define the number of inducing points, here: every 75th point is selected
        ].clone()
        self.inducing_points = torch.nn.Parameter(inducing_points)
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            self.base_covar_module,
            inducing_points=inducing_points,
            likelihood=likelihood,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()


def main(
    fileName,
    height_filtering,
    height_lower,
    height_upper,
    invert_selection,
    decrease_trials,
    decrease_trials_size,
    decrease_duration,
    decrease_duration_size,
    project_name,
    add_prop_features,
    target,
    create_trial_identifier,
    var_thresholding,
    var_threshold,
    testdata_size,
    random_int,
    scalewith,
    n_cross_val,
    npseed,
    dB,
    add_white_noise,
    learning_rate,
    earlystoppingpatience,
    miniumimprovement,
    max_epochs,
    scheduler_patience,
):
    np.random.seed(npseed)
    ############ DATA LOADING AND PREPROCESSING ############
    data = pd.read_csv(fileName, sep=",")

    if (
        height_filtering and invert_selection
    ):  # watch out, this can lead to new all zero columns
        selection = (data.loc[:, "RightHandZ"] > height_lower) & (
            data.loc[:, "RightHandZ"] < height_upper
        )
        data = data[~selection]
    elif (
        height_filtering and not invert_selection
    ):  # watch out, this can lead to new all zero columns
        selection = (data.loc[:, "RightHandZ"] > height_lower) & (
            data.loc[:, "RightHandZ"] < height_upper
        )
        data = data[selection]

    if decrease_trials:
        trial_ids = data.loc[:, "Trial_ID"].values
        unique_trials = np.unique(trial_ids)
        n_trials = unique_trials.size
        n_trials = round(decrease_trials_size * n_trials)
        random_trials = np.random.choice(
            unique_trials, n_trials, replace=False
        )  # set random seed !!!!!
        data = data[data["Trial_ID"].isin(random_trials)]

    if decrease_duration:
        trial_ids = data.loc[:, "Trial_ID"].values
        unique_trials = np.unique(trial_ids)
        mask = []
        for trial in unique_trials:
            trial_indices = data[data["Trial_ID"] == trial].index
            trial_length = len(trial_indices)
            mask.extend(trial_indices[: int(decrease_duration_size * trial_length)])
        data = data.loc[mask]

    total_datapoints = data.shape[0]
    print(f"Total number of datapoints: {total_datapoints}")

    # Split the data into features and target
    X = data.loc[:, "sensor1":"active_sensors"].values
    if add_prop_features:
        X = pd.concat(
            [X, data.loc[:, "GunAX":"GunJZ"].values], axis=1
        )  # adds the xsens prop features to X data
    y = data.loc[:, target].values

    trial_ids = data.loc[:, "Trial_ID"].values

    if create_trial_identifier:
        X = addtrialidentifier(X, trial_ids)

    if var_thresholding:  # deletes features with low variance
        sel = VarianceThreshold(threshold=var_threshold)
        X = sel.fit_transform(X)

    n_groups = np.unique(
        trial_ids
    ).size  # get the number of unique trial ids, which is the number of groups
    n_test_groups = round(testdata_size * n_groups)

    # Initialize GroupShuffleSplit
    gss = GroupShuffleSplit(
        n_splits=1, test_size=n_test_groups, random_state=random_int
    )

    # Split the data
    for train_index, test_index in gss.split(X, y, groups=trial_ids):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        trial_ids_train = data.iloc[train_index, 0].values
        trial_ids_test = data.iloc[test_index, 0].values

    # Check for trial leakage in train/test split
    data_leakage(trial_ids, train_index, test_index)

    # add white noise to the data
    if add_white_noise:
        X_train = whitenoise(X_train, dB)
        X_test = whitenoise(X_test, dB)

    # Standardize the data
    scaler_x = set_standardizer(
        scalewith
    )  # minmax when the features are on the same scale, standard scaler when they are not
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)

    # define the float type, float 64 is more precise than float 32, float 32 is faster
    X_test = torch.tensor(
        X_test, dtype=torch.float64
    )  # train and val tensors are handeled below
    y_test = torch.tensor(y_test, dtype=torch.float64)

    # Early Stopping Configuration
    early_stopping_patience = earlystoppingpatience
    miniumum_delta = miniumimprovement
    best_val_loss = float("inf")
    patience_counter = 0
    initial_patience = early_stopping_patience

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
                "learning_rate": learning_rate,
                "add_white_noise": add_white_noise,
                "SNR": dB,
                "epoch": max_epochs,
                "FYI": "The saved model is the best model according to the lowest validation loss during training.",
                "VarianceThreshold": var_thresholding,
                "variance_threshold": var_threshold,
                "height_preprocess": height_filtering,
                "height_lower": height_lower,
                "height_upper": height_upper,
            },
        )
        config = wandb.config

        X_train_val, X_val = X_train[train_index], X_train[val_index]
        y_train_val, y_val = y_train[train_index], y_train[val_index]

        X_train_val = torch.tensor(X_train_val, dtype=torch.float64)
        y_train_val = torch.tensor(y_train_val, dtype=torch.float64)
        X_val = torch.tensor(X_val, dtype=torch.float64)
        y_val = torch.tensor(y_val, dtype=torch.float64)

        data_leakage(trial_ids_train, train_index, val_index)

        # Reinitialize model and likelihood for each fold
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPModel(X_train_val, y_train_val, likelihood)

        model.train()
        likelihood.train()

        # Optimizer and MLL
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=scheduler_patience
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Early Stopping Initialization
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, config.epoch):
            model.train()
            likelihood.train()
            optimizer.zero_grad()
            output = model(X_train_val)
            loss = -mll(output, y_train_val)
            loss.backward()
            optimizer.step()

            # this is to dynamically change the early stopping patience after the initial convergence, otherwise training takes too long
            if epoch == 50:
                early_stopping_patience = 12
                # the following line resets the patience counter if just before epoch 50 a new best val loss was found or lets it potentially stop at 51 if for already a longer time no new improvement was recorded, in unsure calculate for yourself
                patience_counter = patience_counter - (
                    initial_patience - patience_counter
                )
                if patience_counter < 0:
                    patience_counter = 0

            # Evaluate on validation set
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                val_output = model(X_val)
                val_loss = -mll(val_output, y_val).item()  # Validation loss

                scheduler.step(val_loss)

                # Early Stopping Logic
                if val_loss < best_val_loss:
                    if best_val_loss - val_loss < miniumum_delta:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "likelihood_state_dict": likelihood.state_dict(),
                        },
                        "model_state.pth",
                    )
                    best_inducing_points = model.covar_module.inducing_points.clone()

                    # This block is for testing the model on the test set during training
                    """t = model(X_test)
                    tl = -mll(t, y_test).item()
                    tp = likelihood(model(X_test)).mean
                    tr = torch.sqrt(torch.mean((tp - y_test) ** 2)).item()
                    tr2 = r2_score(y_test.numpy(), tp.numpy())
                    print(f"Epoch {epoch} - Test Loss: {tl:.4f}, Test RMSE: {tr:.4f}, Test R2: {tr2:.4f}")"""
                else:
                    patience_counter += 1

                print(
                    f"Epoch {epoch} - Loss: {loss.item():.4f} Val Loss: {val_loss:.4f} Length Scale: {model.base_covar_module.base_kernel.lengthscale.item():.4f} Ouput Scale: {model.base_covar_module.outputscale.item():.4f} Noise: {likelihood.noise.item():.4f} LR: {scheduler.get_last_lr()[0]}"
                )

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

            model.covar_module._clear_cache()

        # Reload the model and likelihood
        model.covar_module.base_kernel._clear_cache()
        model.base_covar_module._clear_cache()

        state_dict = torch.load("model_state.pth")
        wandb.save("model_state.pth")

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPModel(X_train_val, y_train_val, likelihood)

        model.load_state_dict(state_dict["model_state_dict"], strict=True)
        likelihood.load_state_dict(state_dict["likelihood_state_dict"], strict=True)
        model.covar_module.inducing_points = torch.nn.Parameter(best_inducing_points)

        likelihood.train()  # set to train to clear cache before eval
        model.train()

        model.eval()
        likelihood.eval()

        print(f"Lengthscale: {model.base_covar_module.base_kernel.lengthscale.item()}")
        print(f"Outputscale: {model.base_covar_module.outputscale.item()}")
        print(f"Noise: {model.likelihood.noise.item()}")
        wandb.log(
            {"Lengthscale": model.base_covar_module.base_kernel.lengthscale.item()}
        )
        wandb.log({"Outputscale": model.base_covar_module.outputscale.item()})
        wandb.log({"Noise": model.likelihood.noise.item()})

        # Evaluate on validation and test set
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            voutput = model(X_val)
            vloss = -mll(voutput, y_val).item()
            toutput = model(X_test)
            tloss = -mll(toutput, y_test).item()
            wandb.log({"Val Loss": vloss})
            wandb.log({"Test Loss": tloss})
            test_losses.append(tloss)
            val_losses.append(vloss)

            val_pred = likelihood(model(X_val)).mean
            test_pred = likelihood(model(X_test)).mean

            val_rmse = torch.sqrt(torch.mean((val_pred - y_val) ** 2)).item()
            val_r2 = r2_score(y_val.numpy(), val_pred.numpy())
            test_rmse = torch.sqrt(torch.mean((test_pred - y_test) ** 2)).item()
            test_r2 = r2_score(y_test.numpy(), test_pred.numpy())
            val_mae = mean_absolute_error(y_val.numpy(), val_pred.numpy())
            test_mae = mean_absolute_error(y_test.numpy(), test_pred.numpy())
            wandb.log({"Test RMSE": round(test_rmse, 2)})
            wandb.log({"Test R2 score": round(test_r2, 2)})
            wandb.log({"Test MAE": round(test_mae, 2)})
            wandb.log({"Val RMSE": round(val_rmse, 2)})
            wandb.log({"Val R2 score": round(val_r2, 2)})
            wandb.log({"Val MAE": round(val_mae, 2)})
            val_rmses.append(val_rmse)
            val_R2_scores.append(val_r2)
            val_maes.append(val_mae)
            test_rmses.append(test_rmse)
            test_R2_scores.append(test_r2)
            test_maes.append(test_mae)

            plot1, plot2, plot3 = plot_y(y_test, test_pred, trial_ids_test, target)
            wandb.log({"Actual vs Predicted Values for test set": plot1})
            wandb.log({"Residual Plot": plot2})
            wandb.log({"Actual and Predicted Values line plot": plot3})

            signal_variance = torch.var(y_train_val)
            noise = model.likelihood.noise.item()
            Signal_to_Noise = signal_variance / noise
            print("Signal to Noise Ratio: ", Signal_to_Noise.item())
            wandb.log({"Signal to Noise Ratio": Signal_to_Noise.item()})

            print(
                f"Fold {fold} - Val RMSE: {val_rmse:.4f}, Val R2: {val_r2:.4f}, Val Loss: {vloss:.4f}"
            )
            print(
                f"Fold {fold} - Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}, Test Loss: {tloss:.4f}"
            )

        fold += 1
        wandb.finish()

    # Calculate average validation loss and RMSE
    avg_val_loss = np.mean(val_losses)
    avg_val_rmse = np.mean(val_rmses)
    avg_test_loss = np.mean(test_losses)
    avg_test_rmse = np.mean(test_rmses)
    avg_val_R2_score = np.mean(val_R2_scores)
    avg_test_R2_score = np.mean(test_R2_scores)
    avg_val_mae = np.mean(val_maes)
    avg_test_mae = np.mean(test_maes)

    # The average metrics in table format
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
    # save the script
    wandb.save("sgpr_torch.py")
    wandb.finish()


if __name__ == "__main__":
    main()
