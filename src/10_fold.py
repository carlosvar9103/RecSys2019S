import pandas as pd
import sys
import numpy as np


np.random.seed(1428)
df = pd.read_csv('preprocessed.csv', dtype="str")

msk = np.random.rand(len(df)) <= 0.95
# Split the data into train and test

train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_1.csv' ,   index=False)
test.to_csv('preprocessed_test_1.csv',   index=False)

msk = np.random.rand(len(df)) <= 0.85
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_2.csv' ,   index=False)
test.to_csv('preprocessed_test_2.csv',   index=False)

msk = np.random.rand(len(df)) <= 0.75
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_3.csv' ,   index=False)
test.to_csv('preprocessed_test_3.csv',   index=False)

msk = np.random.rand(len(df)) <= 0.65
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_4.csv' ,   index=False)
test.to_csv('preprocessed_test_4.csv',   index=False)

msk = np.random.rand(len(df)) <= 0.55
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_5.csv' ,   index=False)
test.to_csv('preprocessed_test_5.csv',   index=False)

msk = np.random.rand(len(df)) <= 0.45
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_6.csv' ,   index=False)
test.to_csv('preprocessed_test_6.csv',   index=False)

msk = np.random.rand(len(df)) <= 0.35
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_7.csv' ,   index=False)
test.to_csv('preprocessed_test_7.csv',   index=False)

msk = np.random.rand(len(df)) <= 0.25
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_8.csv' ,   index=False)
test.to_csv('preprocessed_test_8.csv',   index=False)

msk = np.random.rand(len(df)) <= 0.15
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_9.csv' ,   index=False)
test.to_csv('preprocessed_test_9.csv',   index=False)

msk = np.random.rand(len(df)) <= 0.05
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_10.csv' ,   index=False)
test.to_csv('preprocessed_test_10.csv',   index=False)




sys.exit()
