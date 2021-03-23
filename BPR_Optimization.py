# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:39:11 2021

@author: lpott
"""
import argparse
import numpy as np
import pandas as pd
import os
from time import time
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchvision import transforms,utils
import torch

from metrics import *
from utils import *
from preprocessing import *
from datasets import *
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, help='Number of Training Epochs', default=25)
parser.add_argument('--alpha', type=float, help='Learning Rate', default=0.001)
parser.add_argument('--embed_dim',type=int,help="Size of embedding dimension for matrix factorization",default=64)
parser.add_argument('--user_min',type=int,help='The approximate minimum number of items each user must have watched',default=10)
parser.add_argument('--item_min',type=int,help='The approximate minimum number of users each item must have',default=10)
parser.add_argument('--write_filename',type=str,help='The filename to write all the Netflix data to, and later read',default="")
parser.add_argument('--read_filename',type=str,help='The filename to read all the Netflix data from for the Dataframe',default="data.csv")
parser.add_argument('--batch_size',type=int,help='The batch size for stochastic gradient descent',default=128)
parser.add_argument('--reg',type=float,help='The regularization strength on l2 norm',default = 0.0005)
parser.add_argument('--basepath',type=str,help="The basepath to where the Netflix .txt data files are help",default="archive")

args = parser.parse_args()

write_filename = args.write_filename
read_filename = args.read_filename
basepath = args.basepath

user_min = args.user_min
item_min = args.item_min

batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.alpha
reg = args.reg

embedding_dimension = args.embed_dim


print("="*10,"Creating csv file","="*10)
if write_filename != "":
    create_csv(write_filename,basepath)
    
print("="*10,"Creating DataFrame","="*10)
Netflix_df = create_df(read_filename)

print("="*10,"Creating Subsample DataFrame","="*10)
Netflix_filtered_df = filter_df(Netflix_df,user_min,item_min) # Not Quite Right Yet
Netflix_df_reset = reset_df(Netflix_filtered_df)

print("="*10,"Creating Train/Test DataFrames","="*10)
Netflix_train,Netflix_test = train_test_split(Netflix_df_reset)
n_users,n_items = Netflix_train.nunique()

bpr_data = BPR_Dataset(Netflix_train,n_users,n_items)
bpr_dl = DataLoader(bpr_data,batch_size=batch_size,shuffle=True)

print("="*10,"Creating AUC Metric Class","="*10)
criterion = BPR_Loss()
auc_metric = AUC(Netflix_train,Netflix_test,n_users,n_items)

print("="*10,"Creating Matrix Factorization Model","="*10)
model = MF_BPR(n_users,n_items,embedding_dimension).cuda()

optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=reg)

print("="*10,"Starting Gradient Descent","="*10)
for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(tqdm(bpr_dl,position=0, leave=True), 0):
        model.train()
        # get the inputs; data is a list of [inputs, labels]
        uid,iid_positive,iid_negative = data
        
        uid = uid.cuda()
                
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        positive_scores = model(uid,iid_positive.cuda())
        negative_scores = model(uid,iid_negative.cuda())
        
        
        loss = criterion(positive_scores, negative_scores)
        
        loss.backward()
        
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            print("Test AUC: {:.4f}".format(auc_metric(model)))

    
    #print("Test AUC: {:.4f}".format(auc_metric(model)))

print('Finished Training')