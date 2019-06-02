import pandas as pd
import csv
#import matplotlib.pyplot as plt
#import sklearn as sl
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.impute import SimpleImputer
import numpy as np
import random

random.seed(3333)

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

def missing_unknown(row):
    if row == "":
        return None
    else:
        return row

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

def get_data_pure():

    ds = pd.read_csv(input_path+'item_metadata.csv',dtype="str", sep=",", encoding="utf-8")
    return ds

def smaller_data_set(ds,samples):
    rr = random.sample(range(0,ds.shape[0]),samples)
    ds = ds.iloc[rr,:]
    return ds

def generate_ds_sparse(ds,p_unique):

    item_id = ds.item_id
    index = ds.index
    zeros = np.zeros((len(item_id), len(p_unique)), dtype=int)
    zeros = pd.DataFrame(zeros,index=index)
    zeros = pd.concat([item_id,zeros], axis=1)
    p_unique.insert(0,'item_id')
    columns = p_unique
    zeros.columns = columns
    #print(columns)
    return zeros

def fill_ds_sparse(ds_sparse,properties):
    ds_sparse = ds_sparse
    #for row in ds_sparse:
    #    for k in properties:
    #        break
    return ds_sparse


def get_data_preprocessed():

    print("reading raw data..")
    ds = get_data_pure()

    ###JUST TO SEE IF THIS WORKS###
    samples = round(len(ds)*0.001)
    print(samples)
    ds = smaller_data_set(ds,samples)
    #####END####

    index = ds.index
    items = ds.item_id
    #print(ds)


    #x = ds.set_index('item_id').properties.str.split('|', expand=True).stack().reset_index(level=1, drop=True).to_frame('properties')
    #x = ds.set_index('item_id').properties.str.split('|', expand=True).stack()
    x = ds.set_index('item_id', drop=False, append=True).properties.str.split('|', expand=True).stack()
    #x = ds.groupby(level=0).apply(lambda group: pd.Series(group.values.ravel().tolist()[0].split('|')))
    #print(x.head(100))
    print ("creating sparse matrix..")

    #pd.get_dummies(x, prefix='p', columns=['properties']).groupby(level=0).sum()
    #xx = pd.get_dummies(x, prefix=None, prefix_sep=None).groupby(level=0).agg(max)#sum()
    ds_sparse_matrix = pd.get_dummies(x, prefix=None, prefix_sep=None).groupby(level=1, sort=False).agg(max)#sum()
    #itemxx = xx.index
    ds_sparse_matrix.index = index
    ds_sparse_matrix['item_id'] = items
    columns = ds_sparse_matrix.columns.tolist()
    del columns[-1]
    columns.insert(0,'item_id')
    #print(columns)
    ds_sparse_matrix = ds_sparse_matrix[columns]

    print (ds_sparse_matrix.head(10))

    #items_id = ds.item_id
    #properties = ds.properties.str.split("|")
    #properties2 = dict(zip(items_id,properties)) #in case

    #p = properties.str.split("|")
    #p_unique = []
    #meta_sparse = generate_ds_sparse(ds,p_unique)

    ds_sparse_matrix.to_csv(target_path+'item_metadata_sparse.csv', sep=',', index=False)
    return "done"

def main():

    print(get_data_preprocessed())
    exit()


if __name__ == "__main__":
    main()
