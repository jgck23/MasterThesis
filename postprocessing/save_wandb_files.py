import wandb
import os
from tqdm import tqdm

api = wandb.Api()
runs = api.runs("ursja-karlsruhe-institute-of-technology/Test_Windows")
downloadfolder='/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/WandB_Downloads/Test_Windows'

# Build a list of all files with associated run and download folder
all_files = []
for run in runs:
    print(f"Run: {run.name}")
    run_folder = os.path.join(downloadfolder, run.name)
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    for file in run.files():
        all_files.append((run, file, run_folder))

# Download each file with a progress bar
for run, file, run_folder in tqdm(all_files, desc="Downloading files", total=len(all_files)):
    print(f"  Downloading {file.name}")
    file.download(root=run_folder, replace=True)
