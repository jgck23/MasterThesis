import wandb
import pandas as pd

# Initialize API
api = wandb.Api()

# Replace with your entity (username or team) and project name
entity = "ursja-karlsruhe-institute-of-technology"
project = "250318_Eule3_TrialNum"

# Get all runs in the project
runs = api.runs(f"{entity}/{project}")

# Extract config and run info
run_data = []
for run in runs:
    # Start with basic info
    run_info = {
        "name": run.name,
        "id": run.id,
        "created_at": run.created_at,
        "state": run.state,
    }

    # Extract config params (excluding system-generated ones)
    for k, v in run.config.items():
        if not k.startswith("_") and isinstance(v, (int, float, str, bool)):  
            run_info[f"{k}"] = v  

    # Extract summary metrics (excluding W&B artifacts)
    for k, v in run.summary.items():
        if isinstance(v, (int, float, str, bool)):  # Skip dictionaries and lists
            run_info[f"{k}"] = v  

    run_data.append(run_info)

# Convert to DataFrame
df = pd.DataFrame(run_data)

# Save to CSV
df.to_csv("250318_Eule3_TrialNum_NN.csv", index=False)
