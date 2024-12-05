import scipy.io
import glob
import pandas as pd
import tensorflow as tf

def load_data(userinput=None):
    while True:
        try:
            if userinput is None:
                userinput= input("Enter the name of the file you want to create a model for : ")
            # Load the .mat file
            mat_file = glob.glob('**/'+str(userinput)+'.mat', recursive=True)
            if len(mat_file) > 1:
                raise ValueError("Multiple files found")

            mat = scipy.io.loadmat(mat_file[0]) 
            # Print the keys of the dictionary
            #print(mat.keys())
            # Assuming the data is stored in a variable named 'data' in the .mat file
            data = mat['learnerMatrix']
            # Convert the data to a pandas DataFrame
            df = pd.DataFrame(data)
            return df, userinput
        except FileNotFoundError:
            print("File not found")
            pass
        except ValueError:
            print("Invalid input")
            pass

def set_optimizer(optimizer, learning_rate=0.001, beta_1=0.9, beta_2=0.999):
    optimizer = optimizer.lower().strip()
    if optimizer == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    elif optimizer == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False)
    elif optimizer == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
    elif optimizer == "adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer == "adadelta":
        return tf.keras.optimizers.Adadelta(learning_rate=learning_rate, rho=0.95)
    elif optimizer == "adamax":
        return tf.keras.optimizers.Adamax(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    elif optimizer == "nadam":
        return tf.keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    elif optimizer == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    else:
        raise ValueError("Invalid optimizer")
    
def set_regularizer(regularizer, l1=0.01, l2=0.01):
    regularizer = regularizer.lower().strip()
    if regularizer == "l1":
        return tf.keras.regularizers.L1(l1)
    elif regularizer == "l2":
        return tf.keras.regularizers.L2(l2)
    elif regularizer == "l1_l2":
        return tf.keras.regularizers.L1L2(l1, l2)
    else:
        raise ValueError("Invalid regularizer")
    
def data_leakage(trial_ids, train_index, test_index):
    train_trials = set(trial_ids[train_index])
    test_trials = set(trial_ids[test_index])
    common_trials = train_trials.intersection(test_trials)
    if common_trials:
        return ValueError(f"Trial leakage detected between train and test sets for trials: {common_trials}")