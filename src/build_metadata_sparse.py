import pandas as pd
import numpy as np
import csv
import random

random.seed(3333)

input_path = "data_raw/"
target_path = "data_prepro/"


def get_data_pure():
    ds = pd.read_csv(input_path+'item_metadata.csv',dtype="str", sep=",", encoding="utf-8")
    return ds

def smaller_data_set(ds,samples):
    rr = random.sample(range(0,ds.shape[0]),samples)
    ds = ds.iloc[rr,:]
    return ds

def get_data_preprocessed():

    print("reading raw data..")
    ds = get_data_pure()

    ###JUST USE THIS TO WORK WITH A SMALLER item_metadata.csv DATASET###
    samples = round(len(ds)*0.001)
    print(samples)
    ds = smaller_data_set(ds,samples)
    #####END#####

    index = ds.index
    items = ds.item_id
    #print(ds)

    x = ds.set_index('item_id', drop=False, append=True).properties.str.split('|', expand=True).stack()
    #print(x.head(100))

    print ("creating sparse matrix..")
    ds_sparse_matrix = pd.get_dummies(x, prefix=None, prefix_sep=None).groupby(level=1, sort=False).agg(max)
    #itemsxx = ds_sparse_matrix.index
    ds_sparse_matrix.index = index
    ds_sparse_matrix['item_id'] = items
    columns = ds_sparse_matrix.columns.tolist()
    del columns[-1]
    columns.insert(0,'item_id')
    #print(columns)
    ds_sparse_matrix = ds_sparse_matrix[columns]

    print (ds_sparse_matrix.head(10), ds_sparse_matrix.shape)

    ds_sparse_matrix.to_csv(target_path+'item_metadata_sparse.csv', sep=',', index=False)

    return "done"

def main():
    print(get_data_preprocessed())
    exit()


if __name__ == "__main__":
    main()
