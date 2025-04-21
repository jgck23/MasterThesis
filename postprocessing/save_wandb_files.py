# this file is used to save the wandb files from a project into a local folder
import wandb
import os

api = wandb.Api()
runs = api.runs("ursja-karlsruhe-institute-of-technology/250312_Pferd12_Chained")
downloadfolder = "//ipek.kit.edu/dfs/Messdaten/Messdaten_8TB/4828_Jubot_Forcebased_Posture_Estimation/MA_Sembrizki/pferd12/WandB/Chained_NN"

# Iterate through all runs
for run in runs:
    unique_run_name = f"{run.name}_{run.id}"  # Ensure uniqueness
    run_folder = os.path.join(downloadfolder, unique_run_name)

    if not os.path.exists(run_folder):
        os.makedirs(run_folder)

    print(f"Downloading files for Run: {run.name} (ID: {run.id}) into {run_folder}")

    # Download all files for this run
    for file in run.files():
        print(f"  Downloading {file.name}")
        file.download(root=run_folder, replace=True)
