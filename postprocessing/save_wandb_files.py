import wandb
import os
from tqdm import tqdm

api = wandb.Api()
runs = api.runs("ursja-karlsruhe-institute-of-technology/Test_Windows")
downloadfolder = "/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/WandB_Downloads/Test_Windows"

# Iterate through all runs
for run in tqdm(runs, desc="Processing Runs"):
    unique_run_name = f"{run.name}_{run.id}"  # Ensure uniqueness
    run_folder = os.path.join(downloadfolder, unique_run_name)

    if not os.path.exists(run_folder):
        os.makedirs(run_folder)

    print(f"Downloading files for Run: {run.name} (ID: {run.id}) into {run_folder}")

    # Download all files for this run
    for file in run.files():
        print(f"  Downloading {file.name}")
        file.download(root=run_folder, replace=True)
