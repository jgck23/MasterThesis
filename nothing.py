import csv
import numpy as np
file_path = '/Users/jacob/Documents/Microsoft Visual Studio Code Projects/Masterarbeit/Data/Foot Sensor Force Data/241113_Leopard24.csv'

data = []
with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        data.append(row)
data = np.array(data)
print(data)