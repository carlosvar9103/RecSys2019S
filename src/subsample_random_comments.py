import pandas as pd
import csv
#import matplotlib.pyplot as plt
import sklearn as sl
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import random

input_path = "data_raw/"
target_path = "data_prepro/"

def count_param(row, param):
    print(row)
    return row.lower().count(param)

def label(row, param):
    return param[int(row)]

def is_class(row, param):
    return int(int(row) == param)

def is_class_str(row, param):
    return int(str(row) == param)

def is_equal_str(row, param):
    return int(str(row) == str(param))

def is_present(row):
    return int(int(row)>0)

def nan_unknown(row):
    if row == "?":
        return None
    else:
        return row

def missing_unknown(value):
    if value == "":
        return None
    else:
        return value

def yes_no_unknown(row):
    if row == "y":
        return 1
    elif row == "n":
        return 0
    else:
        return None

def randon_list(max,ds):
    randoms = []
    for n in range(max):
        r = random.randint(0,ds.shape[0])
    return randoms


def read_special_list(string):
    lst = string.split("||")
    lst2 = []
    for x in lst:
        x = x[x.rfind(":")+1:]
        lst2.append(x)
    return lst2

def get_data_pure(name):
    df = pd.read_csv(input_path+name, skipinitialspace=True,dtype="str", sep=",", encoding="utf-8")#.replace('"','', regex=True)#quotcechar='"',delimiter="\n",quoting=csv.QUOTE_ALL, engine="python"
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
