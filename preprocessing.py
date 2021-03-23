# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:34:27 2021

@author: lpott
"""
import numpy as np
import pandas as pd
import os
from time import time
from sklearn.preprocessing import LabelEncoder

def create_csv(filename="data.csv",netflix_basepath="../Datasets/archive/"):
    data = open('data.csv', mode='w')
    
    files = [os.path.join(netflix_basepath,file) for file in 
             ['combined_data_1.txt','combined_data_2.txt','combined_data_3.txt','combined_data_4.txt']]
    
    t_start= time()
    for file in files:
        print("Opening file: {}".format(file))
        with open(file) as f:
            for line in f:
                line = line.strip()
                if line.endswith(':'):
                    movie_id = line.replace(':', '')
                else:
                    data.write(movie_id + ',' + line)
                    data.write('\n')
    t_end = time()
    print("{:.2f} Seconds".format(t_end-t_start))
    data.close()
    
def create_df(filename="data.csv"):
    # Read all data into a pd dataframe
    
    t_start = time()
    df = pd.read_csv(filename, names=['item_id', 'user_id'],usecols=[0,1])    
    df = df.reindex(columns=['user_id','item_id'])
    t_end = time()
    
    print(df.nunique())
    print(df.shape)
    print("{:.2f} Seconds".format(t_end-t_start))
    
    return df.reset_index(drop=True)

def filter_df(df=None,user_min=10,item_min=10):
    if df is None:
        return 
    
    t_start = time()
    
    user_counts = df.groupby('user_id').size()
    user_subset = np.in1d(df.user_id,user_counts[user_counts >= item_min].sample(10000).index)
    
    filter_df = df[user_subset].reset_index(drop=True)
    
    # find items with 10 or more users
    item_counts = filter_df.groupby('item_id').size()
    item_subset = np.in1d(filter_df.item_id,item_counts[item_counts >= user_min].sample(5000).index)    
    
    filter_df = filter_df[item_subset].reset_index(drop=True)
    
    # cannot have user ids with less than 5...
    user_counts = filter_df.groupby('user_id').size()
    user_subset = np.in1d(filter_df.user_id,user_counts[user_counts >= 5].index)
    
    filter_df = filter_df[user_subset].reset_index(drop=True)
    
    
    t_end = time()

    
    assert (filter_df.groupby('user_id').size() < 5).sum() == 0
    assert (filter_df.groupby('item_id').size() < 5).sum() == 0
    
    
    print(filter_df.nunique())
    print(filter_df.shape)
    print("{:.2f} Seconds".format(t_end-t_start))
    
    return filter_df

def reset_df(df=None):
    item_enc = LabelEncoder()
    df['item_id'] = item_enc.fit_transform(df['item_id'])
    
    user_enc = LabelEncoder()
    df['user_id'] = user_enc.fit_transform(df['user_id'])
    
    assert df.user_id.min() == 0
    assert df.item_id.min() == 0 
    
    return df
    