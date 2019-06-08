import pandas as pd
import numpy as np
import csv
import random

random.seed(3333)

input_path = "data_prepro/"
target_path = "data_prepro/"


def get_data_pure():
    ds = pd.read_csv(input_path+'train_sub_train.csv',dtype="str", sep=",", encoding="utf-8")
    return ds

def smaller_data_set(ds,samples):
    rr = random.sample(range(0,ds.shape[0]),samples)
    ds = ds.iloc[rr,:]
    return ds

def get_data_preprocessed():

    print("reading raw data..")
    ds = get_data_pure()
    print(ds.head(10), ds.shape)

    ###JUST USE THIS TO WORK WITH A SMALLER item_metadata.csv DATASET###
    #samples = round(len(ds)*0.1)
    #print(samples)
    #ds = smaller_data_set(ds,samples)
    #####END#####

    index = ds.index
    users = ds.user_id
    session_id = ds.session_id
    tiemstamps = ds.timestamp
    ds['session_id_timestamp'] = ds[['session_id', 'timestamp']].apply(lambda x: ''.join(x), axis=1)
    ds['user_id_timestamp'] = ds[['user_id', 'timestamp']].apply(lambda x: ''.join(x), axis=1)
    ds.current_filters = ds.current_filters.fillna("No filter")
    print(ds.head(10), ds.shape)


    print ("creating sparse matrix..")
    dm_action_type = pd.get_dummies(ds.action_type, prefix=None, prefix_sep=None)#.groupby(level=1, sort=False).agg(max)
    print(dm_action_type.head(10), dm_action_type.shape)
    ds = pd.concat([ds,dm_action_type], axis=1)
    print(ds.head(10), ds.shape)

    #Alles GUT exit()

    x = ds.set_index('session_id_timestamp', drop=False, append=True).current_filters.str.split('|', expand=True).stack()
    #x = ds.set_index('user_id_timestamp', drop=False, append=True).current_filters.str.split('|', expand=True).stack()

    #print(x.head(1000), x.shape)

    dm_current_filters = pd.get_dummies(x, prefix=None, prefix_sep=None).groupby(level=0, sort=False).agg(max)

    print(dm_current_filters.head(10), dm_current_filters.shape)

    #ds = pd.concat([ds,dm_action_type], axis=1)
    ds = pd.concat([ds,dm_current_filters], axis=1)

    print(ds.head(10), ds.shape)


    ds.to_csv(target_path+'sub_train_sparse.csv', sep=',', index=False)



    # exit()
    # #x = ds.set_index('item_id', drop=False, append=True).action_type.stack()
    # #print(x.head(100))
    #
    # print ("creating sparse matrix..")
    # ds_sparse_matrix = pd.get_dummies(x, prefix=None, prefix_sep=None).groupby(level=1, sort=False).agg(max)
    # #itemsxx = ds_sparse_matrix.index
    # ds_sparse_matrix.index = index
    # ds_sparse_matrix['item_id'] = items
    # columns = ds_sparse_matrix.columns.tolist()
    # del columns[-1]
    # columns.insert(0,'item_id')
    # #print(columns)
    # ds_sparse_matrix = ds_sparse_matrix[columns]
    #
    # print (ds_sparse_matrix.head(10), ds_sparse_matrix.shape)
    #
    # ds_sparse_matrix.to_csv(target_path+'sub_item_metadata_sparse.csv', sep=',', index=False)

    return "done"

def main():
    get_data_preprocessed()
    print ("done")
    exit()


if __name__ == "__main__":
    main()
