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

#define default weights for item matching:
#parameters:
max_weight = 2
w_prices = 3

w_ci = 5 #clickout action
w_iimg = 1 #interaction item image
w_iid = 1 #interaction item deals
w_iir = 2 #interaction item rating
w_iiin= 2 #interaction item info
w_sfi = 3 #search for item


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
    print(metadata.head(5))
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
    print(X.groupby(['session_id']).impressions.last())
    last_imp = pd.DataFrame(X.groupby(['session_id']).impressions.last(),columns=['impressions'])
    last_imp_sample = last_imp.copy()
    last_imp.reset_index(level=0, inplace=True)
    print(last_imp.head(10),last_imp.shape)
    #temp1 = create_impressions(X) NOT NEEDED I THINK
    num_imp = last_imp[['impressions']].applymap(lambda x: str.count(x, '|')).impressions.tolist()
    print(num_imp[:10],np.shape(num_imp))
    hola = input("Hola: ")
    #= number of impressions per session_id (since not all sessions actually have 25 items in the impressions list)
    sess_list= impressions.index.tolist()
    ytrain = pd.DataFrame(np.zeros((len(sess_list),  max(num_imp)+1))) #create empty matrix for y train (only zeros)
    ytrain.set_index(last_imp["session_id"], inplace=True) #set indices as the session_id's

    #fill in the for ytrain values accordingly:
    pbar = ProgressBar()

    for i in pbar(last_imp.index.values.tolist()): #for all the values in the last_imp matrix #pbar(range(0,len(sess_list))):
        if sampling==False:
            ytrain.iloc[i,0:(num_imp[i]+1)]=np.linspace(1,0,num=num_imp[i]+1)
        else:
            #probs = np.linspace(0.99,0.01,num=num_imp[i]+1)# still need to transform so that they sum up to 1
            probs = np.logspace(1, 0, num=num_imp[i]+1, endpoint=True, base=1000.0)
            np.random.seed(i)
            #ytrain.iloc[i,0:(num_imp[i]+1)]=np.linspace(1,0,num=num_imp[i]+1)
            ytrain.iloc[i,0:(num_imp[i]+1)]=np.random.choice(np.linspace(1,0,num=num_imp[i]+1), size=num_imp[i]+1, replace=False, p=probs/probs.sum())
    return ytrain

def build_extend_train(df): #create train with dummies from action_type and current_filters
    index = df.index
    users = df.user_id
    session_id = df.session_id
    timestamps = df.timestamp
    df['session_id_step'] = df[['session_id', 'step']].apply(lambda x: ''.join(x), axis=1) #only unique combination
    df.current_filters = df.current_filters.fillna("No filter")
    dm_action_type = pd.get_dummies(df.action_type, prefix=None, prefix_sep=None) #create dummy matrix for column "action_type"
    df2 = pd.concat([df,dm_action_type], axis=1) #add information about action_types

    #split up "current_filters":
    x = df2.set_index('session_id_step', drop=False, append=True).current_filters.str.split('|', expand=True).stack()
    dm_current_filters = pd.get_dummies(x, prefix=None, prefix_sep=None).groupby(level=0, sort=False).agg(max)

    out = pd.concat([df2,dm_current_filters], axis=1)
    return out

def create_one_row_per_sess(df):
    pbar = ProgressBar()

    colnames= df.columns.values.tolist()[14:] #element 13... change of sort order => add manually at the beginning
    temp = pd.DataFrame(df.groupby(['session_id'])["change of sort order"].max(),columns=["change of sort order"])
    #with session_id as index
    for i in pbar(range(0,len(colnames))):
        addit = pd.DataFrame(df.groupby(['session_id'])[colnames[i]].max(),columns=[colnames[i]]) #with session_id as index
        temp = pd.merge(temp, addit, left_on=["session_id"], right_on=["session_id"]) #merge them by session_id (=index)
    return temp

def weights_for_references(df,impressions,w_ci=w_ci,w_iimg=w_iimg,w_iir=w_iir,w_iiin=w_iiin,w_sfi=w_sfi): #default weights
    pbar = ProgressBar() #use Progess bar for loop

    out = pd.DataFrame().reindex_like(impressions).fillna(value=0) # Create a flexible matrix full of zeros where there is no values, just to finish it as an output
    print(out.head(20))
    input("Hola mundo: ")
    sessions = impressions.index.values.tolist() #all session_id's
    temp = df[["session_id","action_type","reference"]] # create a matrix temp with just the columns session_id, action_type and references
    #The idea here is to weight the values based on the implicit feedback received by the user's behaviour, the values are:
    # w_ci = 5 #clickout action
    # w_iimg = 1 #interaction item image
    # w_iid = 1 #interaction item deals
    # w_iir = 2 #interaction item rating
    # w_iiin= 2 #interaction item info
    # w_sfi = 3 #search for item

    for sess in pbar(sessions):
        #we create lists for every single type of interaction and save the list in arrays: hit_* for each type of interaction
        hit_clickout= temp.loc[(temp["session_id"] == sess) & (temp["action_type"] == "clickout item")].reference.values.tolist()
        hit_iimg =  temp.loc[(temp["session_id"] == sess) & (temp["action_type"] == "interaction item image")].reference.values.tolist()
        hit_iir =  temp.loc[(temp["session_id"] == sess) & (temp["action_type"] == "interaction item rating")].reference.values.tolist()
        hit_iiin =  temp.loc[(temp["session_id"] == sess) & (temp["action_type"] == "interaction item info")].reference.values.tolist()
        hit_sfi =  temp.loc[(temp["session_id"] == sess) & (temp["action_type"] == "search for item")].reference.values.tolist()

        #add up the weights for each session_id and item:
        out.loc[sess] = impressions.loc[sess].isin(hit_clickout)*w_ci + impressions.loc[sess].isin(hit_iimg)*w_iimg + impressions.loc[sess].isin(hit_iir)*w_iir + impressions.loc[sess].isin(hit_iiin)*w_iiin + impressions.loc[sess].isin(hit_sfi)*w_sfi
        print(out.loc[sess])
        input("Hola mundo2: ")
    return out

def weights_for_prices(prices,extended_choices, max_weight=max_weight,w_prices=w_prices):
    pbar = ProgressBar()

    out = pd.DataFrame().reindex_like(prices).fillna(value=0)
    out = prices.copy()

    #all sessions in which the user did sort by price
    sess_sort_by_price = extended_choices[extended_choices['Sort by Price']==1].index.tolist() #index = "session_id"

    out= prices.iloc[:,:].rank(axis=1,pct=True,method="max",na_option="keep",ascending=False)*max_weight
    #the lowest price receives the highest weight:
    for sess in pbar(sess_sort_by_price):
        out.loc[sess]*=w_prices
    out.fillna(value=0, inplace=True) #set 0 for missing values (NaNs)
    return out

def weights_for_metadata(df,df_meta,impr):
    pbar = ProgressBar()

    col_df = list(df4)[list(df4).index("search for poi")+1:] #column start AFTER "search for poi"
    col_meta = list(metadata)
    cols = [value for value in col_df if value in col_meta] #columns that exist in BOTH dataframes (=intersection)

    df1 = df.loc[:,cols] #information with one row per session_id
    df1.fillna(value=0,inplace=True) #fill empty values with zero

    df2 = df_meta.loc[:,cols]
    df2.fillna(value=0,inplace=True)#fill empty values with zero

    weights_meta = pd.DataFrame(np.zeros((len(impr.index),  25))) #create empty weight matrix (same form as impressions)
    weights_meta.set_index(impr.index, inplace=True) #with the same indices

    meta_ind = df1.index[df1.sum(axis=1)>0].tolist() #extract all session_id's for which any filters were applied

    pbar = ProgressBar()

    for i in pbar(list(meta_ind)):
        hilf = impr.loc[i].rename('item_id',inplace=True).to_frame()
        hilf.set_index('item_id',inplace=True)
        hilf1 = df2.join(hilf,how='inner',on='item_id') #metadata-attributes from the items in impressions list
        hilf2 = df1.loc[i].to_frame() #filters from the dataset
        weights_meta.loc[i,0:len(hilf1.index)-1]= [val for sublist in hilf1.dot(hilf2).values for val in sublist]
        #plug in matrix-matrix-product for computing similarities to weight-matrix

    return weights_meta









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

    #PREPROSESSING AND MATRIX CONSTRUCTION
    #1
    #train_sample = get_data_pure ('train.csv')
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

    #3.3
    print("Writing "+target_path+'prices_sample.csv')
    #prices_sample = create_prices(df2,sampling=True) #create prices matrix
    prices_sample = get_data_exported('prices_sample.csv')
    print(prices_sample.head(10))
    #prices_sample.to_csv(target_path+'prices_sample.csv', sep=',', index=True)

    #3.4
    print("Writing "+target_path+'ytrain_sample.csv')
    #ytrain_sample=create_ytrain(df2,impr_sample,sampling=True)
    ytrain_sample = get_data_exported('ytrain_sample.csv')
    print(ytrain_sample.shape)
    print(ytrain_sample.head(10))
    #ytrain_sample.to_csv(target_path+'ytrain_sample.csv', sep=',', index=True)

    #4.0
    print("Writing "+target_path+'train_sample_extended.csv')
    #df3 = build_extend_train(df2)
    df3 = get_data_exported('train_sample_extended.csv')
    print("Corresponding gain of dimensionality (column-wise): \n")
    print(df2.shape)
    print(df3.shape)
    print(df3.head(10))
    #df3.to_csv(target_path+'train_sample_extended.csv', sep=',', index=False)



    #5
    #input("columns.values.tolist()[14:]: ")
    #print(df3.columns.values.tolist()[14:]) # all the columns after those that appeared before

    print("Writing "+target_path+'train_one_raw_sess.csv')
    #df4 = create_one_row_per_sess(df3)
    df4 = get_data_exported('train_one_raw_sess.csv')
    print("Corresponding loss of dimensionality: \n")
    print(df3.shape)
    print(df4.shape)
    print(df4.head(5))
    #df4.to_csv(target_path+'train_one_raw_sess.csv', sep=',', index=False)

    #5.1
    weights_references_sample = weights_for_references(df3,impr_sample)
    #weights_references_sample = get_data_exported('weights_references_sample.csv')
    print(weights_references_sample.head(10))
    weights_references_sample.to_csv(target_path+'weights_references_sample.csv', sep=',', index=True)

    #5.2
    weights_prices_sample = weights_for_prices(prices_sample,df4)
    weights_prices_sample = get_data_exported('weights_prices_sample.csv')
    print(weights_prices_sample.head(10))
    weights_prices_sample.to_csv(target_path+'weights_prices_sample.csv', sep=',', index=True)

    #5.3
    weights_metadata_sample = weights_for_metadata(df4,metadata,impr_sample)
    weights_metadata_sample = get_data_exported('weights_metadata_sample.csv')
    print(weights_metadata_sample.head(10))
    weights_metadata_sample.to_csv(target_path+'weights_metadata_sample.csv', sep=',', index=True)

    #5.4
    print("Weights for references:")
    print(weights_references_sample.head(5))
    print("Weights for prices:")
    print(weights_prices_sample.head(5))
    print("Weights for metadata:")
    print(weights_metadata_sample.head(5))

    full_weights_sample = weights_prices_sample + weights_references_sample + weights_metadata_sample
    print("Overview of training data and corresponding target values:")
    print(full_weights_sample.head(5), ytrain_sample.head(5))
    full_weights_sample.to_csv(target_path+'full_weights_sample.csv', sep=',', index=True)

    full_weights_sample = get_data_exported("full_weights_sample.csv")
    full_weights_sample.set_index("session_id",inplace=True)
    ytrain_sample = get_data_exported("ytrain_sample.csv")
    ytrain_sample.set_index("session_id",inplace=True)


    #6 Preprocess TEST dataset

    #reading in test data
    print('\n reading test dataset.. \n')
    test = get_data_exported("sub_test.csv")
    #test = get_data_pure('test.csv')
    test_sample = truncate_df(test,cutoff=1) #whole test data
    full_weights_sample.to_csv(target_path+'full_weights_sample.csv', sep=',', index=True)
    full_weights_sample = get_data_exported("full_weights_sample.csv")
    print (test_sample.head(2),test_sample.shape)

    #export the truncated dataset:
    #test_sample.to_csv(target_path+'test_sample.csv', sep=',', index=False)

    # get to df1:
    #Truncate for last destination:
    print("Processing "+target_path+'test_df1.csv')
    test_df1 = trunc_last_dest(test_sample)
    test_df1.to_csv(target_path+'test_df1.csv', sep=',', index=True)
    test_df1 = get_data_exported("test_df1.csv")
    print("Corresponding loss of dimensionality: \n")
    print(test_sample.shape)
    print(test_df1.shape)

    #get to df2 (alternative version):
    print("Processing "+target_path+'submission_popular.csv')
    sub = get_data_pure('submission_popular.csv')
    sess_sub = sub["session_id"].values #extract from the dummy submission-file which sessions are needed
    #thats easier then going go through all sessions and omitting some if the don't fulfill some kind of condition
    test_df2 = test_df1.loc[test_df1['session_id'].isin(sess_sub)]
    test_df2.to_csv(target_path+'test_df2.csv', sep=',', index=True)
    test_df2 = get_data_exported("test_df2.csv")
    print(test_df2.head(10))

    # create impressions and prices (but WITHOUT SAMPLING!!! since its the test data):
    print("Processing "+target_path+'test_impr.csv')
    test_impr = create_impressions(test_df2,sampling=False)
    test_impr.to_csv(target_path+'test_impr.csv', sep=',', index=True)
    print(test_impr.shape)
    print(test_impr.head(5))

    print("Processing "+target_path+'test_prices.csv')
    test_prices= create_prices(test_df2,sampling=False)
    test_prices.to_csv(target_path+'test_prices.csv', sep=',', index=True)
    print(test_prices.shape)
    print(test_prices.head(5))

    #create df3, so include columns with applied filters:
    print("Processing "+target_path+'test_df3.csv')
    test_df3 = build_extend_train(test_df2)
    test_df3.to_csv(target_path+'test_df3.csv', sep=',', index=True)
    print("Corresponding gain of dimensionality (column-wise): \n")
    print(test_df2.shape)
    print(test_df3.shape)
    print(test_df3.head(10))
    #df3.to_csv(target_path+'test_sample_extended.csv', sep=',', index=False)

    #create df4, so create one row per session_id:
    print("Processing "+target_path+'test_df4.csv')
    test_df4 = create_one_row_per_sess(test_df3)
    test_df4.to_csv(target_path+'test_df4.csv', sep=',', index=True)
    print("Corresponding loss of dimensionality: \n")
    print(test_df3.shape)
    print(test_df4.shape)
    display(test_df4.head(5))

    test_impr = get_data_exported("test_impr.csv")
    test_impr.set_index("session_id",inplace=True)
    print(test_impr.head(5))


    print ("done")
    exit()


if __name__ == "__main__":
    main()
