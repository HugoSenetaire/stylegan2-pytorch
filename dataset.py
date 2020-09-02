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


class SimpleDataset(data.Dataset):
    def __init__(self, path, transform, resolution=256):
        self.folder = path
        self.list_image = os.listdir(self.folder)
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        path = os.path.join(self.folder,self.list_image[i])
        img = Image.open(path)
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
    def __init__(self,
        folder,
        transform,
        image_size,
        columns = ["sap_sub_function"],
        columns_inspirationnal = [],
        dataset_type = "unique",
        multiview = False,
        csv_path = None,
        transparent = False):


        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.columns = columns
        self.columns_inspirationnal = columns_inspirationnal
        self.dataset_type = dataset_type
        self.multiview = multiview
        if csv_path is not None :
            self.df = pd.read_csv(csv_path)
        else :
            self.df = pd.read_csv(folder+".csv")

        if self.dataset_type == "stellar" :
            self.df_name = self.df.image_id.drop_duplicates()
            if not self.multiview:
                self.df = self.df[self.df.stellar_view.isin(["Front view", "Vue de Face"])]

        # Get one Hot Encoder
        change = True  
        while(change): # This seems like a really bad way to make sure every category has more than 50 items
            change = False
            self.dic = {}
            self.encoder = {}
            self.dic_column_dim = {}
            for column in columns :
                list_possible_value = []
                for k, value in enumerate(self.df[column].unique()):
                    #print(value,type(value))
                    if self.df[column].value_counts()[value] > 50:
                        list_possible_value.append(value)
                self.dic[column] = copy.deepcopy(list_possible_value)
                self.encoder[column] = OneHot(list_possible_value)
                before_len = len(self.df)
                self.df = self.df[self.df[column].isin(self.dic[column])]
                if before_len != len(self.df):
                    change = True
                self.dic_column_dim[column] = len(self.dic[column])


            self.dic_inspirationnal = {}
            self.dic_column_dim_inspirationnal = {}
            for column in self.columns_inspirationnal :
                list_possible_value = []
                for k, value in enumerate(self.df[column].unique()):
                    if self.df[column].value_counts()[value] > 50:
                        list_possible_value.append(value)
                self.dic_inspirationnal[column] = copy.deepcopy(list_possible_value)
                self.encoder[column] = OneHot(list_possible_value)
                before_len = len(self.df)
                self.df = self.df[self.df[column].isin(self.dic_inspirationnal[column])]
                if before_len != len(self.df):
                    change = True
                self.dic_column_dim_inspirationnal[column] = len(self.dic_inspirationnal[column])
 
        print("Total lenght of remaining dataframe")
        print(len(self.df))

        for column in columns :
            print(f"Saved value for column {column}")
            print(self.df[column].value_counts())
        for column in self.columns_inspirationnal:
            print(f"Saved value for column {column}")
            print(self.df[column].value_counts())

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _create_one_hot(self, data, columns, dic):
        dic_label = {}
        if len(columns)>0 :
            x = []
            aux = self.encoder[columns[0]].apply(data[columns[0]])
            one_hot = torch.zeros(len(dic[columns[0]])).scatter_(0, torch.tensor([aux]), 1.0)
            dic_label[columns[0]] = aux
            for i,column in enumerate(columns):
                if i == 0 :
                    continue
                aux = self.encoder[column].apply(data[column])
                dic_label[column] = aux
                one_hot = torch.cat((one_hot,torch.zeros(len(dic[column])).scatter_(0, torch.tensor([aux]), 1.0)))


            x = one_hot.type(torch.float32)
        else :
            x = -1

        return x, dic_label

    def __getitem__(self, index):
        if self.dataset_type == "stellar" :
            data = self.df.iloc[index]
            name = data.image_id
            url = data.akamai_asset_link.split("/")[-1].replace(" ","%20")
            path = os.path.join(self.folder,url)
        else :
            data = self.df.iloc[index]
            name = data.image_id
            path = os.path.join(self.folder,name+".jpg")

        # TODO Very bad way to deal with the problem of the dataset
        img = Image.open(path).convert('RGB').resize((self.image_size,self.image_size))
        img_transform = self.transform(img)
    
        x,dic_label = self._create_one_hot(data, self.columns,self.dic)
        y,dic_inspiration = self._create_one_hot(data, self.columns_inspirationnal, self.dic_inspirationnal)

        if len(self.columns)>0:
            if len(self.columns_inspirationnal)>0 :
                x = torch.cat([x,y])
            return x, img_transform, dic_label, dic_inspiration

        elif len(self.columns_inspirationnal)>0:
            return y, img_transform, dic_label, dic_inspiration
            
        else :
            return -1, img_transform, dic_label, dic_inspiration
   

        
    
    def get_len(self, type = None):
        size = 0
        if type is None or type == "label":
            for column in self.columns:
                size+=len(self.dic[column])
        if type is None or type == "inspirationnal":
            for column in self.columns_inspirationnal:
                size+=len(self.dic_inspirationnal[column])

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

    ### Sampling methods
    
    def random_one_hot(self,batch_size):
        if len(self.columns)==0 :
            return None
        
        one_hot = torch.zeros((batch_size,self.get_len(type="label")))
        dic_label = {}
        for k in range(batch_size):
            previous_size = 0
            for i,column in enumerate(self.columns):
                aux = np.random.randint(len(self.dic[column]))
                if k==0 :
                    dic_label[column] = [aux]
                else :
                    dic_label[column].append(aux)
                one_hot[k][aux+previous_size]=1
                previous_size +=len(self.dic[column])
        
        for column in self.columns:
            dic_label[column] = torch.tensor(dic_label[column])

        return one_hot,dic_label


    def listing_one_hot(self,batch_size):
        if len(self.columns) == 0 :
            return None
        
        one_hot = torch.zeros((batch_size,self.get_len(type="label")))
        dic_label = {}
        for k in range(batch_size):
            previous_size = 0
            for i,column in enumerate(self.columns):
                aux = k
                if k==0 :
                    dic_label[column] = [aux]
                else :
                    dic_label[column].append(aux)
                one_hot[k][aux+previous_size]=1
                previous_size +=len(self.dic[column])

        for column in self.columns:
            dic_label[column] = torch.tensor(dic_label[column])
        return one_hot,dic_label

    def random_weights(self,batch_size):
        if len(self.columns_inspirationnal)==0 :
            return None

        totalLen = self.get_len(type="inspirationnal")
        weights = torch.zeros((batch_size,totalLen))
        dic_weights = {}
        for k in range(batch_size):
            previous_size = 0
            for i,column in enumerate(self.columns_inspirationnal):
                possibleLen = len(self.dic_inspirationnal[column])
                aux_weights = torch.zeros((possibleLen,),dtype = torch.float32).new_full((possibleLen,),1./possibleLen)
                weights[k][previous_size:previous_size+possibleLen]= aux_weights
                if k==0 :
                    dic_weights[column] = aux_weights[None,:]
                else :
                    dic_weights[column]= torch.cat([dic_weights[column],aux_weights[None,:]])
                previous_size +=possibleLen

        return weights, dic_weights

    def random_extended_weights(self,batch_size):
        if len(self.columns_inspirationnal)==0 :
            return None

        totalLen = self.get_len(type="inspirationnal")
        weights = torch.zeros((batch_size,totalLen))
        dic_weights = {}
        for k in range(batch_size):
            previous_size = 0
            for i,column in enumerate(self.columns_inspirationnal):
                possibleLen = len(self.dic_inspirationnal[column])
                nbFuzzy = np.random.randint(1,possibleLen) 
                fuzzyTaken = np.random.choice(possibleLen, nbFuzzy, replace=False)
                aux_weights = np.zeros((possibleLen,))
                for taken in fuzzyTaken :
                    aux_weights[taken] = 1./nbFuzzy
                aux_weights = torch.tensor(aux_weights, dtype=torch.float32)
                weights[k][previous_size:previous_size+possibleLen]= aux_weights
                if k==0 :
                    dic_weights[column] = aux_weights[None,:]
                else :
                    dic_weights[column]= torch.cat([dic_weights[column],aux_weights[None,:]])
                previous_size +=possibleLen

        return weights, dic_weights


    def sample_manager(self, batch_size, device, label_method = "random", inspiration_method = "full_random"):

        if len(self.columns)>0 :
            if label_method == 'listing':
                sample_label, dic_label = self.listing_one_hot(batch_size)
            elif label_method == 'random' :
                sample_label, dic_label = self.random_one_hot(batch_size)
            else :
                raise(ValueError("Label method not recognized"))
            sample_label = sample_label.to(device)
            for column in self.columns :
                dic_label[column] = dic_label[column].to(device)
            

        if len(self.columns_inspirationnal)>0:
            if inspiration_method == 'fullrandom':
                # sample_weights, dic_weights = self.random_weights(batch_size)
                sample_weights, dic_weights = self.random_extended_weights(batch_size)
            else :
                raise(ValueError("Inspiration method not recognized"))
            sample_weights = sample_weights.to(device)
            for column in self.columns_inspirationnal:
                dic_weights[column] = dic_weights[column].to(device)


        if len(self.columns)>0:
            if len(self.columns_inspirationnal)>0:
                output = torch.cat([sample_label,sample_weights],dim=1)
                return output,dic_label,dic_weights
            else :
                output = sample_label
                return output,dic_label,None
        elif len(self.columns_inspirationnal)>0:
            output = sample_weights
            return output,None,dic_weights
        else :
            return None,None,None

## Manage category for exploration of latent space :
    def category_manager(self, batch_size, device, label_list = None, label_inspiration_list = None, label_method = "random", inspiration_method = "full_random"):
        # if label_list is None :
            # raise Exception("Label should be given to control the category")

        if len(self.columns)>0 :
            if label_list is not None :
                one_hot_label = self.create_label_one_hot(label_list,batch_size=batch_size)
            else :
                if label_method == 'listing':
                    sample_label, dic_label = self.listing_one_hot(batch_size)
                elif label_method == 'random' :
                    sample_label, dic_label = self.random_one_hot(batch_size)
                else :
                    raise(ValueError("Label method not recognized"))
                one_hot_label = sample_label

        if len(self.columns_inspirationnal)>0:
            if label_inspiration_list is not None :
                one_hot_weights = self.create_inspiration_weights(label_inspiration_list, batch_size).to(device)
            else :
                sample_weights, dic_weights = self.random_weights(batch_size)
                sample_weights = sample_weights

        if len(self.columns)>0:
            if len(self.columns_inspirationnal)>0:
                output = torch.cat([one_hot_label,one_hot_weights],dim=1)
                return output.to(device)
            else :
                output = one_hot_label
                return output.to(device)
        elif len(self.columns_inspirationnal)>0:
            output = one_hot_weights
            return output.to(device)
        else :
            return None

    def create_label_one_hot(self, label_list,batch_size = 1):
        nb_columns = len(self.columns)
        if len(label_list) != batch_size or len(label_list[0])!= nb_columns :
            raise Exception("list of label do not have the right size")

        aux = label_list[0][0]
        one_hot = torch.zeros(len(self.dic[self.columns[0]])).scatter_(0, torch.tensor([aux]), 1.0)
        for i,column in enumerate(self.columns):
            if i == 0 :
                continue

            aux = label_list[0][i]
            one_hot = torch.cat((one_hot,torch.zeros(len(self.dic[column])).scatter_(0, torch.tensor([aux]), 1.0)))
            
        one_hot = one_hot[None,:]

        for k in range(1,batch_size):
            if k == 0 :
                continue
            aux = label_list[k][0]
            aux_one_hot = torch.zeros(len(self.dic[self.columns[0]])).scatter_(0, torch.tensor([aux]), 1.0)
            
          
            for i,column in enumerate(self.columns_inspirationnal):
                if i == 0 :
                    continue
                aux = label_list[k][i]
                aux_one_hot = torch.cat((one_hot,torch.zeros(len(self.dic[column])).scatter_(0, torch.tensor([aux]), 1.0)))

            one_hot = torch.cat((one_hot,aux_one_hot[None,:]),dim=0)

        return one_hot


    def create_inspiration_weights(self, label_list,batch_size = 1):
        nb_columns = len(self.columns)
        if len(label_list) != batch_size or len(label_list[0])!= nb_columns :
            raise Exception("list of label do not have the right size")
        
        raise NotImplementedError
