import pandas as pd
import numpy as np
import sys

input_path = "data_prepro/"
target_path = "data_prepro/"
name_train = "sub_train.csv"

df = pd.read_csv(input_path+name_train,dtype="str", sep=",", encoding="utf-8", skipinitialspace=True)

msk = np.random.rand(len(df)) <= 0.90 # Split the data into train and dev

train = df[msk]
dev = df[~msk]
print(train.shape)
print(dev.shape)
train.to_csv(target_path+'train_sub_train.csv' ,   index=False)
dev.to_csv(target_path+'dev_sub_train.csv',   index=False)

'''
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
'''

sys.exit()
