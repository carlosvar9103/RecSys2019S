import pandas as pd
import numpy as np
import csv
import random

random.seed(3333) #fixed seed for random consistency

input_path = "data_raw/"
target_path = "data_prepro/"

def get_data_pure(name):
    df = pd.read_csv(input_path+name,dtype="str", sep=",", encoding="utf-8", skipinitialspace=True)
    return df

def smaller_data_set(ds,samples):
    rr = random.sample(range(0,ds.shape[0]),samples) #get random indexes
    ds = ds.iloc[rr,:] #slide full dataset
    return ds

def get_data_preprocessed():

    print('reading train..')
    ds_train = get_data_pure('train.csv')

    print('reading test..')
    ds_test = get_data_pure('test.csv')

    samples_train = round(ds_train.shape[0]*0.001) #1% of the full dataset
    ds_train = smaller_data_set(ds_train,samples_train)

    samples_test = round(ds_test.shape[0]*0.001) #1% of the full dataset
    ds_test = smaller_data_set(ds_test,samples_test)

    print (ds_train.head(5),ds_train.shape)
    print (ds_test.head(5),ds_test.shape)

    ds_test.to_csv(target_path+'sub_test.csv', sep=',', index=False)
    ds_train.to_csv(target_path+'sub_train.csv', sep=',', index=False)

    return "done"

def main():
    print(get_data_preprocessed())
    exit()


if __name__ == "__main__":
    main()
