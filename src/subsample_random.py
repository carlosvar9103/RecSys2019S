import pandas as pd
import numpy as np
import csv
import random

input_path = "data_raw/"
target_path = "data_prepro/"

def get_data_pure(name):
    df = pd.read_csv(input_path+name, skipinitialspace=True,dtype="str", sep=",", encoding="utf-8")
    return df

def smaller_data_set(ds,samples):
    rr = random.sample(range(0,ds.shape[0]),samples)
    ds = ds.iloc[rr,:]
    return ds

def get_data_preprocessed():
    sample_num = 3333

    ds_train = get_data_pure('train.csv')
    ds_test = get_data_pure('test.csv')

    ds_train = smaller_data_set(ds_train,sample_num)
    ds_test = smaller_data_set(ds_test,sample_num)

    print (ds_train.head(5),ds_train.shape)
    
    print (ds_test.head(5),ds_test.shape)

    ds_test.to_csv(target_path+'sub_test.csv', sep=',', index=False)
    ds_train.to_csv(target_path+'sub_train.csv', sep=',', index=False)
    print("done")

    return sample_num

def main():

    get_data_preprocessed()
    exit()


if __name__ == "__main__":
    main()
