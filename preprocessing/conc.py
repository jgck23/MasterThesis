from fun import load_data
import pandas as pd
# script to concatenate files for the same user
get_names = input("Enter the name of the files you want to concatenate: ")
names = get_names.split()
#print(names)

num_trials = []
for name in names:
    data, name_mat = load_data(name) # load the data, ignore the name_mat
    num_trials.append(data.iloc[-1,0]) # count the number of trials from each file
    if name == names[0]:
        df = data # initialize the dataframe with the first file
    else:
        data.iloc[:,0] = data.iloc[:,0] + sum(num_trials[:-1]) # update the trial ids only if it is not the first file
        df = pd.concat([df, data])

print(num_trials)
df.to_csv('EHKL.csv', index=False)

# update the trial ids in the first column 
# count trials

