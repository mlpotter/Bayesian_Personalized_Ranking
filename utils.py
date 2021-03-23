# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:36:16 2021

@author: lpott
"""
import numpy as np
import pandas as pd
from time import time

def train_test_split(df=None):
    if df is None:
        return
    
    t_start = time()
    
    grouped = df.groupby('user_id')
    
    df_test = grouped.sample(1)
    
    idx_keep = np.where(pd.merge(df,df_test,how='left',indicator='exist').exist == 'both',False,True)
    
    df_train = df[idx_keep]
    
    t_end = time()
    
    print("{:.2f} Seconds".format(t_end-t_start))
    print("Train Set Tuples: ",df_train.shape)
    print("Test Set Tuples: ",df_test.shape)

    return df_train.reset_index(drop=True),df_test.reset_index(drop=True)
