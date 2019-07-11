import pandas as pd
import numpy as np
import csv
import random
import sys
import os
import multiprocessing


random.seed(3333)

input_path = "data_raw/"
target_path = "data_prepro/"
flip_path = "data_flip/"


def get_data_pure(name,input_path=input_path):
    ds = pd.read_csv(input_path+name,dtype="str", sep=",", encoding="utf-8")
    return ds

def get_data_exported(name,target_path=target_path):
    df = pd.read_csv(target_path+name,dtype="str", sep=",", encoding="utf-8", skipinitialspace=True)
    return df

def truncate_df(input,cutoff=0.02): #use per default 2% of the data
    random.seed(3333)
    sessions=input['session_id'].unique().tolist()
    np.random.seed(3333)
    if cutoff==1: #use entire data and keep the order:
        sess_rand = sessions
    else: #cut the data and randomly order it
        np.random.seed(3333)
        sess_rand = np.random.choice(sessions, size=int(np.floor(cutoff*len(sessions))), replace=False) #random sample of sessions without replacement
    output= input.loc[input['session_id'].isin(sess_rand)]
    return output

def smaller_data_set(ds,samples):
    rr = random.sample(range(0,ds.shape[0]),samples)
    ds = ds.iloc[rr,:]
    return ds

def create_metadata(name,cutoff=1): #use per default the whole metadata informations
    ds= get_data_pure(name)
    np.random.seed(3333)
    samples = round(len(ds)*cutoff) # use a specific ratio of the metadata (per default use all)
    ds2 = smaller_data_set(ds,samples) #truncate data
    x = ds2.set_index('item_id', drop=False, append=True).properties.str.split('|', expand=True).stack()
    #split up the data string that is just seperated by "|"s

    print ("creating metadata matrix.. \n")
    metadata = pd.get_dummies(x, prefix=None, prefix_sep=None).groupby(level=1, sort=False).agg(max)
    print(metadata.shape)
    display (metadata.head(5))
    metadata.to_csv(target_path+'metadata.csv', sep=',', index=True)
    return metadata



def get_data_preprocessed():
    print("reading raw data..")
    name = 'train.csv'
    #name = 'train.csv'
    train = get_data_pure(name)
    #truncate the data to 33% of the size (for finaltraining run):
    train_sample = truncate_df(train,cutoff=0.02)
    print("In the process, the dataset got truncated to " + str(round(100*train_sample.shape[0]/train.shape[0],3)) + " % of the size. \n")
    print("Overall there are " + str(train['session_id'].nunique()) +
          " unique sessions, that's mean an average length of " + str(train.shape[0]/train['session_id'].nunique()) + " per session. \n")
    print (train_sample.head(5),train_sample.shape)

    print ("The train sample has been written with file name"+name)
    name = 'train_sample.csv'
    train_sample.to_csv(target_path+name, sep=',', index=False)
    print ("The train sample has been written with file name: "+ target_path + name)

    print(train_sample.head(5), train_sample.shape)
    return train_sample

def get_data_preprocessed2():

    print("reading raw data..")
    name = 'train.csv'
    #name = 'train.csv'
    train = get_data_pure(name)
    #truncate the data to 33% of the size (for finaltraining run):
    train_sample = truncate_df(train,cutoff=0.02)
    print("In the process, the dataset got truncated to " + str(round(100*train_sample.shape[0]/train.shape[0],3)) + " % of the size. \n")
    print("Overall there are " + str(train['session_id'].nunique()) +
          " unique sessions, that's mean an average length of " + str(train.shape[0]/train['session_id'].nunique()) + " per session. \n")
    print (train_sample.head(5),train_sample.shape)

    print ("The train sample has been written with file name"+name)
    name = 'train_sample.csv'
    train_sample.to_csv(target_path+name, sep=',', index=False)
    print ("The train sample has been written with file name: "+ target_path + name)

    print(train_sample.head(5), train_sample.shape)
    exit()

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

def trunc_last_dest(train_df):

    #first: contruct dataframe with session_id and the last corresponding step (=LAST clickout)
    last_ev = pd.DataFrame(train_df.groupby(['session_id']).step.last(),columns=['step'])
    last_ev.reset_index(level=0, inplace=True) #convert index session_id to an actual column (from pandas library)

    #second: merge with city (= destination at the final clickout) :
    last_city = pd.merge(last_ev, df[["session_id","step","city"]], left_on=["session_id","step"], right_on=["session_id","step"])

    #third: get back the original full dataset (=all columns), but this time only for the rows whose destination is similar
    #to the last destination:
    df_restr = pd.merge(train_df,last_city[["session_id","city"]],on=["session_id","city"])

    #Some useful information
    print("Corresponding loss of dimensionality: \n")
    print("Actual train: "+train_df.shape)
    print("Last Step train: "+df_restr.shape)

    return df_restr


    return

def main():
    train_sample = get_data_preprocessed()
    name_metadata = "item_metadata.csv"
    metadata = create_metadata("item_metadata.csv") #use all available data
    #OPPORTUNITY TO IMPROVEMENT USING THE OTHER STEPS
    train_last_step = trunc_last_dest(train_sample) #use the train_sample and compress it taking the last step



    print ("done")
    exit()


if __name__ == "__main__":
    main()
