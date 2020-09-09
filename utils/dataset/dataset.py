from io import BytesIO
import os
import lmdb
from PIL import Image
import json
from tqdm import tqdm
from math import floor, log2
import random
from shutil import rmtree
from functools import partial
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils import data




def convert_rgb_to_transparent(image):
    if image.mode == 'RGB':
        return image.convert('RGBA')
    return image

def convert_transparent_to_rgb(image):
    if image.mode == 'RGBA':
        return image.convert('RGB')
    return image





class SimpleDataset(data.Dataset):
    def __init__(self, path, transform, resolution=256):
        self.folder = path
        self.list_image = []
        for i,element in enumerate(os.listdir(self.folder)) :
            if element.endswith(".png"):
                self.list_image.append(element)
      
        
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.list_image)

    def __getitem__(self, index):
        path = os.path.join(self.folder,self.list_image[index])
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
        transparent = False,
        transform_mask = None,
        limit_category = None):


        super().__init__()
        self.folder = folder

        self.transform_mask = transform_mask
        self.folder_mask = self.folder + "_masks"
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
                    if limit_category is not None and limit_category != "None" :
                        if len(columns)>1:
                            raise NotImplemented("limit_category should only be used by calc_inception and columns should only be unique")
                        if value != limit_category :
                            continue
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
        self.transform_mask = transform_mask

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

        path_mask = os.path.join(self.folder_mask,name+".jpg")
        if self.transform_mask is not None :
            mask = Image.open(path_mask).convert('L')
            mask_transform = self.transform_mask(mask).unsqueeze(1)
        else :
            mask_transform = -1
    
        x,dic_label = self._create_one_hot(data, self.columns,self.dic)
        y,dic_inspiration = self._create_one_hot(data, self.columns_inspirationnal, self.dic_inspirationnal)


        if len(self.columns)>0:
            if len(self.columns_inspirationnal)>0 :
                x = torch.cat([x,y])
            return x, img_transform, dic_label, dic_inspiration, mask_transform

        elif len(self.columns_inspirationnal)>0:
            return y, img_transform, dic_label, dic_inspiration, mask_transform
            
        else :
            return -1, img_transform, dic_label, dic_inspiration, mask_transform
   

        
    
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
    
 
    def get_onehot_fromvalue(self,value):
        if len(self.columns)>1 :
            raise ValueError("This is only to be used in test FID with one column")
        column = self.columns[0]
        onehot_index = self.encoder[column].apply(value)
        return onehot_index

    def upscale(self):  
        self.image_size = self.image_size*2
        transform = transforms.Compose(
            [   
                transforms.Lambda(convert_transparent_to_rgb),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((self.image_size,self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        self.transform = transform
        if self.transform_mask is not None :
            self.transform_mask = transform_mask
            transform_mask = transforms.Compose(
                    [
                        transforms.Resize((self.image_size, self.image_size)),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
                    ]
                )

    

    def get_reverse(self,index):
        if len(self.columns)==0 :
            return None
        liste = []
        for i,column in enumerate(self.columns):
            liste.append(self.encoder[column].reverse(index[i]))
            
        return liste

    ### Noise creation :

    def make_noise(self,batch, latent_dim, n_noise, device):
        if n_noise == 1:
            return torch.randn(batch, latent_dim, device=device)

        noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

        return noises



    def make_zero_noise(self,batch, latent_dim, n_noise, device):
        if n_noise == 1:
            return torch.zeros(batch, latent_dim, device=device)

        noises = torch.zeros(n_noise, batch, latent_dim, device=device).unbind(0)

        return noises



    def mixing_noise(self,batch, latent_dim, prob, device, zero = False):
        if not zero :
            if prob > 0 and random.random() < prob:
                return self.make_noise(batch, latent_dim, 2, device)

            else:
                return [self.make_noise(batch, latent_dim, 1, device)]
        else :
            if prob > 0 and random.random() < prob:
                return self.make_zero_noise(batch, latent_dim, 2, device)

            else:
                return [self.make_zero_noise(batch, latent_dim, 1, device)]    

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
                aux = k%len(self.dic[column])
                print()
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
                if np.random.randint(2)>0 :
                    nbFuzzy = np.random.randint(1,possibleLen+1) 
                else :
                    nbFuzzy = 1
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

        
    def random_mask(self,batch_size, return_name = False):
        list_name = []
        index = np.random.randint(0,len(self.df))
        data = self.df.iloc[index]
        name = data.image_id
        path_mask = os.path.join(self.folder_mask,name+".jpg")
        mask = Image.open(path_mask).convert('L')
        mask_transform = self.transform_mask(mask).unsqueeze(1)
        total = mask_transform
        list_name.append(name)
        for i in range(batch_size-1):
            index = np.random.randint(0,len(self.df))
            data = self.df.iloc[index]
            name = data.image_id
            path_mask = os.path.join(self.folder_mask,name+".jpg")
            mask = Image.open(path_mask).convert('L')
            mask_transform = self.transform_mask(mask).unsqueeze(1)
            total = torch.cat([total,mask_transform],dim=0)
            list_name.append(name)
        if return_name :
            return total,list_name
        return total

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
                one_hot_weights, dic_weights = self.random_weights(batch_size)

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


        if batch_size!= len(label_list) :
            batch_size = len(label_list)
        if len(label_list) != batch_size or len(label_list[0])!= nb_columns :
            raise Exception("list of label do not have the right size")

        one_hot = torch.zeros((batch_size,self.get_len(type="label")))
        for k in range(batch_size):
            previous_size = 0
            for i,column in enumerate(self.columns):
     
        
                aux = label_list[k][i]
                one_hot[k][aux+previous_size]=1
                previous_size +=len(self.dic[column])


        return one_hot




    def create_inspiration_weights(self, label_dic,batch_size = 1):
        nb_columns = len(self.columns)

     

        if len(label_dic) != batch_size :
            raise Exception("list of label do not have the right size")

        if len(self.columns_inspirationnal)==0 :
            return None

        totalLen = self.get_len(type="inspirationnal")
        weights = torch.zeros((batch_size,totalLen))
        dic_weights = {}
        for k in range(batch_size):
            previous_size = 0
            for i,column in enumerate(self.columns_inspirationnal):
                possibleLen = len(self.dic_inspirationnal[column])
                nbFuzzy = len(label_dic[0][column])
                fuzzyTaken = label_dic[0][column]
                aux_weights = np.zeros((possibleLen,))
                for taken in fuzzyTaken :
                    aux_weights[taken] = 1./nbFuzzy
                aux_weights = torch.tensor(aux_weights, dtype=torch.float32)
                weights[k][previous_size:previous_size+possibleLen]= aux_weights
                previous_size +=possibleLen

        return weights
        
