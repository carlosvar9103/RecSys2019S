import pandas as pd
import csv
#import matplotlib.pyplot as plt
import sklearn as sl
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import random

input_path = "data_raw"
target_path = "data_prepro"

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
    #csv.register_dialect('myDialect',delimiter = ',',quoting=csv.QUOTE_ALL,skipinitialspace=True)
    df = pd.read_csv(input_path+'/original_train.csv', skipinitialspace=True,dtype="str", sep=",", encoding="utf-8")#.replace('"','', regex=True)#quotcechar='"',delimiter="\n",quoting=csv.QUOTE_ALL, engine="python"
    return df

def smaller_data_set(ds,samples):
    rr = random.sample(range(0,ds.shape[0]),samples)
    ds = ds.iloc[rr,:]
    return ds

def get_data_preprocessed():
    ds = get_data_pure()
    ds = smaller_data_set(ds,3333)
    #print (ds.head(10),ds.shape)




    #print(ds.tail(10))
    #print (ds.shape[0])
    #print(ds.columns)
    #print(ds)
    #s= ds.columns
    #print(s)
    #features = 'user_id,session_id,timestamp,step,action_type,reference,platform,city,country,device,current_filters,impressions,prices'
    #features2= "user_id,session_id,timestamp,step,action_type,reference,platform,city,device,current_filters,impressions,prices"
    #s.replace('"', '')
    #columns = features.split(",")
    #print(columns)
    #print(ds)
    #ds.columns=columns
    #print(ds.columns)
    #print("columns given\n",ds.head(10))
    #prices = ds.prices#.apply(lambda x: x.split()[0])
    #print (prices)
    #id = ds.user_id
    #ds = ds.drop(["id","carName"], axis=1)
    #cols = ds.columns
    #print (brand_names)
    #ds.brand = brand_names
    #print(ds)

    #ds.loc[ds['brand'] == '', 'brand'] = ds.carName.str.split().str.get(0)
    #col = ds.columns

    #columns = ds.columns
    #for c in columns:
        #ds[c] = ds.apply(lambda row: missing_unknown(row[c]), axis=1)
        #ds[c] = ds[c].astype("float")
        #print (c)

    '''
    carName = ds["carName"]
    carName.str.split().str.get(0)
    clasz = ds["class"]
    id = ds["id"]
    ds = ds.drop(["class", "ID"], axis=1)
    col = ds.columns
    '''

    #col = ds.columns
    #imp = SimpleImputer(strategy="mean", missing_values=np.nan)
    #imp = SimpleImputer(strategy="most_frequent")

    #ds = pd.DataFrame(imp.fit_transform(ds))
    #ds.columns = col


    '''
    print(ds)
    ds = pd.concat([ds,id], axis=1)
    ds = pd.concat([ds, clasz], axis=1)
    print(ds)
    '''
    #ds.columns = columns
    #ds = pd.concat([id,ds,brand_names], axis=1)
    #brands = ds["carName"].unique()
    #for brand in brands:
    #    ds[brand] = ds.apply(lambda row: is_class_str(row["carName"], brand), axis=1)
    #print(ds)
    ds.to_csv(target_path+'/preprocessed.csv', sep=',', index=False)
    print("done")
    return ds

def main():

    get_data_preprocessed()
    exit()


if __name__ == "__main__":
    main()
