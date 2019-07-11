import pandas as pd
import numpy as np
import csv
import random
import sys
import os
import multiprocessing
from multiprocessing import Pool
from progressbar import ProgressBar


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

def create_metadata1(ds,cutoff=1): #use per default the whole metadata informations
    np.random.seed(3333)
    samples = round(len(ds)*cutoff) # use a specific ratio of the metadata (per default use all)
    ds2 = smaller_data_set(ds,samples) #truncate data
    x = ds2.set_index('item_id', drop=False, append=True).properties.str.split('|', expand=True).stack()
    #split up the data string that is just seperated by "|"s

    print ("creating metadata matrix.. \n")
    metadata = pd.get_dummies(x, prefix=None, prefix_sep=None).groupby(level=1, sort=False).agg(max)
    print(metadata.shape)
    print(metadata.head(10))
    #metadata.to_csv(target_path+'metadata.csv', sep=',', index=True)
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

def get_data_preprocessed1():
    print("reading raw data..")
    train_sample = 'train_sample.csv'
    train_sample = pd.read_csv(target_path+train_sample,dtype="str", sep=",", encoding="utf-8")
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
    last_city = pd.merge(last_ev, train_df[["session_id","step","city"]], left_on=["session_id","step"], right_on=["session_id","step"])
    #third: get back the original full dataset (=all columns), but this time only for the rows whose destination is similar
    #to the last destination:
    df_restr = pd.merge(train_df,last_city[["session_id","city"]],on=["session_id","city"])
    #turn it on.. if necessary
    #df_restr.to_csv(target_path+'train_laststep.csv', sep=',', index=True)
    #Some useful information
    print("Corresponding loss of dimensionality: \n")
    print("Actual train: ",train_df.shape)
    print("Last Step train: ",df_restr.shape)

    return df_restr

#Remove sessions where reference=NA or the impressions list is empty for the clickout item:
def clean_data(df):

    #first: contruct dataframe with session_id and the reference at the last corresponding step (=LAST clickout)
    last_ref = pd.DataFrame(df.groupby(['session_id']).reference.last(),columns=['reference'])
    last_ref.reset_index(level=0, inplace=True) #convert index session_id to an actual column

    #second: same for impressions list:
    last_imp = pd.DataFrame(df.groupby(['session_id']).impressions.last(),columns=['impressions'])
    last_imp.reset_index(level=0, inplace=True)

    #third: merge together => columns: sessions_id, reference, impressions
    temp = pd.merge(last_ref, last_imp, left_on=["session_id"], right_on=["session_id"])


    #fourth step: remove irrelevant sessions:
    temp2=temp[temp.reference.apply(lambda x: x.isnumeric())] #drop session if reference value is not a number
    temp3= temp2.dropna(axis=0,subset=['impressions']) #drop session if impressions list is NaN

    #fifth step: get back the original full dataset (=all columns)
    out = pd.merge(df,pd.DataFrame(temp3["session_id"]),on=["session_id"])
    print("Corresponding loss of dimensionality: \n")
    print(df.shape)
    print(out.shape)
    print(out.head(10))

    return out

def create_impressions(X,sampling=False):
    pbar = ProgressBar()

    last_imp = pd.DataFrame(X.groupby(['session_id']).impressions.last(),columns=['impressions']) #take the last impression when groupby session_id
    last_imp_sample = last_imp.copy() #
    last_imp.reset_index(level=0, inplace=True) #dataset for impressions for last clickout
    if sampling==False:
        out = pd.DataFrame(last_imp["impressions"].str.split("|", expand = True))
        out.set_index(last_imp["session_id"], inplace=True)
    else: #In case of using the smaller sample dataset of train
        num_imp = last_imp[['impressions']].applymap(lambda x: str.count(x, '|')).impressions.tolist()
        impress = last_imp["impressions"].str.split("|")
        for i in pbar(last_imp.index.values.tolist()):
            #sample with probabilities (from 0.99 to 0.01 [to avoid a probability of exactly zero])
            #probs = np.linspace(0.99,0.01,num=num_imp[i]+1)# still need to transform so that they sum up to 1
            probs = np.logspace(1, 0, num=num_imp[i]+1, endpoint=True, base=1000.0)
            np.random.seed(i)
            #sample WITHOUT replacement and with the corresponding probabilities
            last_imp_sample.iloc[i] = ('|'.join(np.random.choice(impress[i], size=num_imp[i]+1, replace=False, p=probs/probs.sum())))
        out = pd.DataFrame(last_imp_sample["impressions"].str.split("|", expand = True))
        out.set_index(last_imp["session_id"], inplace=True)
    return out

def create_prices(X,sampling=False):
    pbar = ProgressBar()

    last_imp = pd.DataFrame(X.groupby(['session_id']).prices.last(),columns=['prices'])
    last_imp_sample = last_imp.copy()
    last_imp.reset_index(level=0, inplace=True) #analog with prices
    if sampling==False:
        out = pd.DataFrame(last_imp["prices"].str.split("|", expand = True))
        out.set_index(last_imp["session_id"], inplace=True)
    else:
        num_pric = last_imp[['prices']].applymap(lambda x: str.count(x, '|')).prices.tolist()
        prices = last_imp["prices"].str.split("|")
        for i in pbar(last_imp.index.values.tolist()):
            #sample with probabilities (from 0.99 to 0.01 [to avoid a probability of exactly zero])
            #probs = np.linspace(0.99,0.01,num=num_pric[i]+1)# still need to transform so that they sum up to 1
            probs = np.logspace(1, 0, num=num_pric[i]+1, endpoint=True, base=1000.0)
            np.random.seed(i)
            #sample WITHOUT replacement and with the corresponding probabilities
            last_imp_sample.iloc[i] = ('|'.join(np.random.choice(prices[i], size=num_pric[i]+1, replace=False, p=probs/probs.sum())))
        out = pd.DataFrame(last_imp_sample["prices"].str.split("|", expand = True))
        out.set_index(last_imp["session_id"], inplace=True)
    return out

def create_ytrain(X,impressions,sampling=False):
    pbar = ProgressBar()

    last_imp = pd.DataFrame(X.groupby(['session_id']).impressions.last(),columns=['impressions'])
    last_imp_sample = last_imp.copy()
    last_imp.reset_index(level=0, inplace=True)
    #temp1 = create_impressions(X) NOT NEEDED I THINK
    num_imp = last_imp[['impressions']].applymap(lambda x: str.count(x, '|')).impressions.tolist()
    #= number of impressions per session_id (since not all sessions actually have 25 items in the impressions list)
    sess_list= impressions.index.tolist()
    ytrain = pd.DataFrame(np.zeros((len(sess_list),  max(num_imp)+1))) #create empty matrix for y train (only zeros)
    ytrain.set_index(last_imp["session_id"], inplace=True) #set indices as the session_id's

    #fill in the for ytrain values accordingly:
    pbar = ProgressBar()
    for i in pbar(last_imp.index.values.tolist()): #pbar(range(0,len(sess_list))):
        if sampling==False:
            ytrain.iloc[i,0:(num_imp[i]+1)]=np.linspace(1,0,num=num_imp[i]+1)
        else:
            #probs = np.linspace(0.99,0.01,num=num_imp[i]+1)# still need to transform so that they sum up to 1
            probs = np.logspace(1, 0, num=num_imp[i]+1, endpoint=True, base=1000.0)
            np.random.seed(i)
            #ytrain.iloc[i,0:(num_imp[i]+1)]=np.linspace(1,0,num=num_imp[i]+1)
            ytrain.iloc[i,0:(num_imp[i]+1)]=numpy.random.choice(np.linspace(1,0,num=num_imp[i]+1), size=num_imp[i]+1, replace=False, p=probs/probs.sum())
    return ytrain

def main():

    num_cores = multiprocessing.cpu_count()

    #name_metadata = "item_metadata.csv"
    # meta = get_data_pure(name_metadata)
    #
    # print (len(meta))
    #
    # chunksize = round(len(meta)/num_cores)
    #
    # metadata = create_metadata("item_metadata.csv") #use all available data
    # metadata = create_metadata1(meta) #use all available data
    #
    # print("Parallel processes")
    # with Pool(processes = (num_cores-1)) as pool:
    #     metadata = pool.map(create_metadata1, meta, chunksize)
    #

    #1
    train_sample = get_data_exported('train_sample.csv')
    print("got the train_sample")

    #2
    metadata = get_data_exported("item_metadata_sparse.csv")
    print(metadata.head(10))

    #OPPORTUNITY TO IMPROVEMENT USING THE OTHER STEPS
    #3.1
    print("truncating training to last step..")
    train_last_step = get_data_exported('train_laststep.csv')
    #train_last_step = trunc_last_dest(train_sample) #use the train_sample and compress it taking the last step
    #train_last_step.to_csv(target_path+'train_laststep.csv', sep=',', index=True)
    print(train_last_step.head(10))

    #3.2
    print("cleaning train_last_step..")
    #df2=clean_data(train_last_step)
    df2 = get_data_exported('train_laststep_clean.csv')
    #df2.to_csv(target_path+'train_laststep_clean.csv', sep=',', index=True)
    print(df2.head(10))

    #3.3
    print("Writing "+target_path+'impr_sample.csv')
    #impr_sample = create_impressions(df2,sampling=True) #create impressions matrix
    impr_sample = get_data_exported('impr_sample.csv')
    print(impr_sample.head(10))
    #impr_sample.to_csv(target_path+'impr_sample.csv', sep=',', index=True)

    print("Writing "+target_path+'prices_sample.csv')
    #prices_sample = create_prices(df2,sampling=True) #create prices matrix
    prices_sample = get_data_exported('prices_sample.csv')
    print(prices_sample.head(10))
    #prices_sample.to_csv(target_path+'prices_sample.csv', sep=',', index=True)




    print ("done")
    exit()


if __name__ == "__main__":
    main()
