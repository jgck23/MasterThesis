import pandas as pd

fileName = 'Data/241212_Dataset_Leopard24_IMU.csv'
outputFileName = 'Data/241212_Dataset_Leopard24_IMU_Resampled.csv'
data = pd.read_csv(fileName, sep=",")

uniqueTrials = data['Trial_ID'].unique()
trialLengths = data.groupby('Trial_ID').size()
averageLength = round(trialLengths.mean())
print(f"Average trial length: {round(averageLength,0)} data points")

def pad_in_order(df, target_len):
    L = len(df)
    # how many full repeats of the whole trial, and how many extra rows
    reps, rem = divmod(target_len, L)
    pieces = [df] * reps
    if rem:
        pieces.append(df.iloc[:rem])
    return pd.concat(pieces, ignore_index=True)

# Resample the data so that every trial has the same number of data points
resampledData = []
for trial in uniqueTrials:
    trialData = data[data['Trial_ID'] == trial]
    if trialData.shape[0] >= averageLength:
        # If the trial is longer than the average, truncate it
        trialData = trialData.iloc[:int(averageLength)]
    elif trialData.shape[0] < averageLength:
        # If the trial is shorter than the average, resample it
        trialData = pad_in_order(trialData, int(averageLength))
    resampledData.append(trialData)

# Combine all resampled trials into a single DataFrame
resampledData = pd.concat(resampledData, ignore_index=True)

# Save the resampled data to a new CSV file
resampledData.to_csv(outputFileName, index=False)
print(f"Resampled data saved to {outputFileName}")