# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:41:02 2021

@author: lpott
"""

from torch.utils.data import Dataset
import torch
import numpy as np

class BPR_Dataset(Dataset):
    """Movie Lens User Dataset"""
    
    def __init__(self,data_df,n_users,n_items):

        self.data_df = data_df
        self.n_users = n_users
        self.n_items = n_items
        
        self.user_group = data_df.groupby('user_id')
        
        
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
            
        uid,iid_positive = self.data_df.iloc[idx]
        
        
        while True:
            iid_negative = np.random.randint(0,self.n_items)
            if not iid_negative in self.user_group.get_group(uid).item_id.tolist():
                break
        
        sample = (uid,iid_positive,iid_negative)
        
        return sample