# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:37:25 2021

@author: lpott
"""

import torch.nn as nn
import torch

class MF_BPR(nn.Module):
    def __init__(self,n_users,n_items,embedding_dimension=64):
        super(MF_BPR, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embedding_dimension
        
        self.user_embedding = nn.Embedding(self.n_users,self.embed_dim)
        self.item_embedding = nn.Embedding(self.n_items,self.embed_dim)

    def forward(self, uid, iid):
        
        
        return torch.sum(self.user_embedding(uid)*self.item_embedding(iid),dim=1)