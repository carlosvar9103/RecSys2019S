import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sl
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

def compute_quartal(pd_data_frame, column_name):
    """
    This funciton should change the year from floating point format to user friendly format. For example 2011.241783 should be changed to 2011 Q1 and 2014.75 should be changed to 2014 Q3
    :param pd_data_frame: padas data set to be changed
    :param column_name: name of the column where the year is saved
    :return: nothing. We work on the data referenced.
    """
    pd_data_frame[column_name].astype(str)
    for i in range(0, pd_data_frame[column_name].size):
        if (str(pd_data_frame[column_name][i]).find("Q") >= 0):
            continue

        temp = (float(pd_data_frame[column_name][i]) * 100) % 100
        if (temp < 25):
            pd_data_frame.loc[pd_data_frame[column_name] == pd_data_frame[column_name][i], column_name] = str(
                int(pd_data_frame[column_name][i])) + " Q1"
        elif (temp < 50):
            pd_data_frame.loc[pd_data_frame[column_name] == pd_data_frame[column_name][i], column_name] = str(
                int(pd_data_frame[column_name][i])) + " Q2"
        elif (temp < 75):
            pd_data_frame.loc[pd_data_frame[column_name] == pd_data_frame[column_name][i], column_name] = str(
                int(pd_data_frame[column_name][i])) + " Q3"
        else:
            pd_data_frame.loc[pd_data_frame[column_name] == pd_data_frame[column_name][i], column_name] = str(
                int(pd_data_frame[column_name][i])) + " Q4"


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

def yes_no_unknown(row):
    if row == "y":
        return 1
    elif row == "n":
        return 0
    else:
        return None

def read_special_list(string):
    lst = string.split("||")
    lst2 = []
    for x in lst:
        x = x[x.rfind(":")+1:]
        lst2.append(x)
    return lst2

def get_data_pure():
    return pd.read_csv('data/AutoMPG.shuf.train.csv', dtype="str", sep=",")

def get_data_preprocessed():
    ds = get_data_pure()
    #print(ds)
    #s= ds.columns
    #print(s)
    s = "id,mpg,carName,cylinders,displacement,horsepower,weight,acceleration,modelYear,origin"
    #s.replace('"', '')
    columns = s.split(",")
    #print(columns)
    #print(ds)
    brand_names = ds.carName.apply(lambda x: x.split()[0])
    id = ds.id
    ds = ds.drop(["id","carName"], axis=1)
    cols = ds.columns
    #print (brand_names)
    #ds.brand = brand_names
    #print(ds)

    #ds.loc[ds['brand'] == '', 'brand'] = ds.carName.str.split().str.get(0)
    #col = ds.columns

    columns = ds.columns
    for c in columns:
        ds[c] = ds.apply(lambda row: nan_unknown(row[c]), axis=1)
        ds[c] = ds[c].astype("float")
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
    imp = SimpleImputer(strategy="mean", missing_values=np.nan)
    #imp = SimpleImputer(strategy="most_frequent")

    ds = pd.DataFrame(imp.fit_transform(ds))
    #ds.columns = col


    '''
    print(ds)
    ds = pd.concat([ds,id], axis=1)
    ds = pd.concat([ds, clasz], axis=1)
    print(ds)
    '''
    ds.columns = columns
    ds = pd.concat([id,ds,brand_names], axis=1)
    brands = ds["carName"].unique()
    for brand in brands:
        ds[brand] = ds.apply(lambda row: is_class_str(row["carName"], brand), axis=1)
    ds.to_csv("preprocessed.csv", sep=',', index=False)
    return ds

def main():

    get_data_preprocessed()
    exit()


if __name__ == "__main__":
    main()