from io import BytesIO
import os
import lmdb
from PIL import Image
from torch.utils import data

import json
from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing

import numpy as np
import pandas as pd
import copy

import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F



class MultiResolutionDataset(data.Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img




class OneHot():
    def __init__(self, list_values):
        self.list_values = list_values
        self.dic = {}
        self.rev_dic = {}
        for i,value in enumerate(self.list_values):
            self.dic[value] = i
            self.rev_dic[i] = value
           
        
    def apply(self,value):
        return self.dic[value]
    
    def reverse(self,i):
        return self.rev_dic[i]

class Dataset(data.Dataset):
    #def __init__(self, folder, image_size,columns = ["sap_function"], transparent = False):
    def __init__(self, folder, transform, image_size,columns = ["sap_sub_function"], transparent = False, transform_mask = None):
        super().__init__()
        self.folder = folder
        self.folder_mask = self.folder[:-1] + "_mask"
        self.image_size = image_size
        self.columns = columns
        self.df = pd.read_csv(folder+".csv")
        
        # Get one Hot Encoder
        self.dic = {}
        self.encoder = {}
        for column in columns :
            list_possible_value = []
            for k, value in enumerate(self.df[column].unique()):
                #print(value,type(value))
                if self.df[column].value_counts()[value] > 200:
                    list_possible_value.append(value)
            self.dic[column] = copy.deepcopy(list_possible_value)
            self.encoder[column] = OneHot(list_possible_value)
            self.df = self.df[self.df[column].isin(self.dic[column])]
            print(f"Saved value for column {column}")
            print(list_possible_value)
            
        #convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        #num_channels = 3 if not transparent else 4
        
        self.transform = transform
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self.df.iloc[index]
        name = data.image_id
        path = os.path.join(self.folder,name+".jpg")
        path_mask = os.path.join(self.folder_mask,name+".jpg")
        img = Image.open(path).convert('RGB')
        img_transform = self.transform(img)
        if self.transform_mask is not None :
            mask = Image.open(path_mask).convert('RGB')
            mask_transform = self.transform_mask(mask)
        else :
            mask_transform = None
        if len(self.columns)>0 :
            x = []
            for column in self.columns:
                aux = self.encoder[column].apply(data[column])
                data_year_one_hot = torch.zeros(len(self.dic[column])).scatter_(0, torch.tensor([aux]), 1.0)

            x = data_year_one_hot.type(torch.float32)
        else :
            x = -1
        return x,img_transform,mask_transform
    
    def get_len(self):
        size = 0
        for column in self.columns:
            size+=len(self.dic[column])
        return size
    
    
    def get_onehot(self,value):
        if len(self.columns)==0 :
            return None
        liste = []
        for column in self.columns:
            liste.append(self.encoder[column].apply(data[column]))
        liste = torch.tensor(liste)
        return liste
    
    def get_reverse(self,index):
        if len(self.columns)==0 :
            return None
        liste = []
        for i,column in enumerate(self.columns):
            liste.append(self.encoder[column].reverse(index[i]))
            
        return liste
    
    def random_one_hot(self,batch_size):
        
        if len(self.columns)==0 :
            return None
        aux = np.random.randint(len(self.dic[self.columns[0]]))
        data_year_one_hot = torch.zeros(len(self.dic[self.columns[0]])).scatter_(0, torch.tensor([aux]), 1.0)
        for i,column in enumerate(self.columns):
            if i == 0 :
                continue
            aux = np.random.randint(len(self.dic[column]))
            data_year_one_hot = torch.cat((data_year_one_hot,torch.zeros(len(self.dic[column])).scatter_(0, torch.tensor([aux]), 1.0)))
            
        
        data_year_one_hot = data_year_one_hot[None,:]
        for k in range(batch_size-1):
            aux = np.random.randint(len(self.dic[self.columns[0]]))
            aux_data_year_one_hot = torch.zeros(len(self.dic[self.columns[0]])).scatter_(0, torch.tensor([aux]), 1.0)
            for i,column in enumerate(self.columns):
                if i == 0 :
                    continue

                aux = np.random.randint(len(self.dic[column]))
                aux_data_year_one_hot = torch.cat((data_year_one_hot,torch.zeros(len(self.dic[column])).scatter_(0, torch.tensor([aux]), 1.0)))
            data_year_one_hot = torch.cat((data_year_one_hot,aux_data_year_one_hot[None,:]),dim=0)
        
        return data_year_one_hot


    def listing_one_hot(self,batch_size):
        if len(self.columns)==0 :
            return None
        #aux = np.random.randint(len(self.dic[self.columns[0]]))
        aux = 0
        data_year_one_hot = torch.zeros(len(self.dic[self.columns[0]])).scatter_(0, torch.tensor([aux]), 1.0)
        for i,column in enumerate(self.columns):
            if i == 0 :
                continue
            #aux = np.random.randint(len(self.dic[column]))
            aux = 0
            data_year_one_hot = torch.cat((data_year_one_hot,torch.zeros(len(self.dic[column])).scatter_(0, torch.tensor([aux]), 1.0)))
            
        
        data_year_one_hot = data_year_one_hot[None,:]
        for k in range(1,batch_size):
            aux = k % len(self.dic[self.columns[0]])
            #aux = np.random.randint(len(self.dic[self.columns[0]]))
            aux_data_year_one_hot = torch.zeros(len(self.dic[self.columns[0]])).scatter_(0, torch.tensor([aux]), 1.0)
            for i,column in enumerate(self.columns):
                if i == 0 :
                    continue
                aux = k % len(self.dic[column])
                #aux = np.random.randint(len(self.dic[column]))
                aux_data_year_one_hot = torch.cat((data_year_one_hot,torch.zeros(len(self.dic[column])).scatter_(0, torch.tensor([aux]), 1.0)))
            data_year_one_hot = torch.cat((data_year_one_hot,aux_data_year_one_hot[None,:]),dim=0)
        
        return data_year_one_hot

    def random_mask(self,batch_size):
        index = np.random.randint(0,self.batch_size)
        data = self.df.iloc[index]
        name = data.image_id
        path_mask = os.path.join(self.folder_mask,name+".jpg")
        mask = Image.open(path_mask).convert('RGB')
        mask_transform = self.transform_mask(mask)
        total = mask_transform[None, :, :, :]
        
        for i in range(batch_size-1):
            index = np.random.randint(0,self.batch_size)
            data = self.df.iloc[index]
            name = data.image_id
            path_mask = os.path.join(self.folder_mask,name+".jpg")
            mask = Image.open(path_mask).convert('RGB')
            mask_transform = self.transform_mask(mask)[None,:,:,:]
            total.cat(mask_transform,dim=0)
        
        return total



