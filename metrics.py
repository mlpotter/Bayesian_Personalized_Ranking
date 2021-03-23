# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:36:50 2021

@author: lpott
"""

import torch
import pandas as pd
from time import time
import numpy as np
from tqdm import tqdm 

class BPR_Loss(object):
    def __init__(self):
        pass
        
    def __call__(self,positive_scores,negative_scores):
        return torch.log( 1 + torch.exp(-positive_scores+negative_scores) ).sum()
    
class AUC(object):
    def __init__(self,train_df,test_df,n_users,n_items):
        
        t_start = time()
        self.total_user_group = pd.concat((train_df,test_df)).reset_index(drop=True).groupby('user_id')
        self.test_df = test_df.set_index(['user_id'])
        self.n_users = n_users
        self.n_items = n_items
        self.all_items = np.arange(self.n_items)
        
        #self.no_click = {uid: torch.LongTensor([idx for idx in self.all_items if idx not in df.item_id.values]) for uid,df in tqdm(self.total_user_group)}
        self.no_click = {uid: torch.LongTensor(list(set.difference(set(self.all_items.tolist()),set(df.item_id.values.tolist())))) for uid,df in tqdm(self.total_user_group,position=0, leave=True)}
        t_end = time()
        
        print("Time to initialize: {:.2f}".format(t_end-t_start))
        
    def __call__(self,model):
        model.eval()
        with torch.no_grad():
            running_auc = 0
            
            user_ids = torch.arange(self.n_users).cuda()
            item_idx = torch.arange(self.n_items).cuda()
            
            for uid in tqdm(user_ids,position=0, leave=True):
                # user id 
                user_id = uid.unsqueeze(0)
                
                # test item from test set
                positive_iid = torch.LongTensor([self.test_df.loc[uid.item()].item()]).cuda()
                
                # score for user id and test item from test set
                positive_score = model.forward(user_id,positive_iid)

                
                # negative score for user id and negative items not in train set or test set (user did not click on)
                filtered_idx = self.no_click[uid.item()].cuda()
                negative_scores = model.forward(user_id.repeat(self.n_items)[filtered_idx],item_idx[filtered_idx])
        
                # AUC score for specific user
                running_auc += ( (positive_score>negative_scores).sum().item() / len(filtered_idx) )
                
        return running_auc/self.n_users