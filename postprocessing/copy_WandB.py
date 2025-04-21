# script similar to https://github.com/wandb/wandb/issues/5726 
# this script is used to copy runs from one wandb project to another
import wandb
import json

# Set your API key
wandb.login()

# Set the source and destination projects
src_entity = "ursja-karlsruhe-institute-of-technology"
src_project = "250318_Eule3_Split_SGPR"
dst_entity = "ursja-karlsruhe-institute-of-technology"
dst_project = "250318_Eule3_TrialNum_SGPR"

# Initialize the wandb API
api = wandb.Api()

# Get the runs from the source project
runs = api.runs(f"{src_entity}/{src_project}")

# Iterate through the runs and copy them to the destination project

for run in runs:
    # Get the run history and files
    history = run.history()
    files = run.files()

    # Create a new run in the destination project
    new_run = wandb.init(project=dst_project, entity=dst_entity, config=run.config, name=run.name,resume="allow")
    
    # Log the history to the new run
    for index, row in history.iterrows():
        new_run.log(row.to_dict())

    # Upload the files to the new run
    for file in files:
        file.download(replace=True)
        new_run.save(file.name,policy = "now")

    for key, value in run.summary.items():
        try:
            json.dumps(value) 
            new_run.summary[key] = value
        except TypeError:
            print(f"Skipping non-serializable summary key: {key}")

    # Finish the new run
    new_run.finish()