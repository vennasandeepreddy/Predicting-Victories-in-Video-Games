import pandas
import pickle
import numpy as np

table_path = "C:/Users/syeda/PycharmProjects/PredictingVictories/raw data/training_data_tabular_format.csv"

dataSet = {}
with open('flattened_preprocessed_data.pkl', 'rb') as f:
    dataSet = pickle.load(f)

df = pandas.read_csv(table_path)

for i in range(len(df['id'])):
    dataSet[str(df['id'][i])][0] = df['winner'][i]

with open('training_data.pkl', 'wb') as f:
    pickle.dump(dataSet, f)
