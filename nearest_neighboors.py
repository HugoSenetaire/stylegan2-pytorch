import argparse
import pickle
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception3
import numpy as np
from tqdm import tqdm

from inception import InceptionV3
# from dataset import MultiResolutionDataset
from dataset import Dataset,SimpleDataset
from utils import *
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import copy



# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class CNN_final(nn.Module):
    def __init__(self, normalization, cnn):
        super(CNN_final, self).__init__()
        self.normalization = normalization
        self.cnn = cnn




    def forward(self, img):
        img = self.normalization(img)
        img = self.cnn(img)
        return img

def create_cnn() :
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context    
    cnn = models.vgg19(pretrained=True).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)

    
    cnn_final = CNN_final(normalization,cnn)
    return cnn_final 






def extract_features(cnn, loader, device):
    pbar = tqdm(loader)

    feature_list = []
    
      
    for img in pbar:
        if isinstance(img,list):
            img = img[1]
        img = img.to(device)
        feature = cnn.forward(img)
        feature = feature.flatten(1)
        feature_list.append(feature.detach())

    features = torch.cat(feature_list, 0).cpu()

    return features


def findNearestNeighboors(inputFeature, testFeature, nbNeighboor = 5):
    listeClosest = []
    listDistance = []
    for i in range(len(testFeature)):
        testedFeature = testFeature[i]
        print(np.shape(testedFeature))
        print(np.shape(inputFeature))
        dist = np.subtract(inputFeature, testedFeature)**2
        closest = np.argsort(dist)[:nbNeighboor]
        dist = np.sort(dist)[:nbNeighboor]
        listeClosest.append(closest)
        listDistance.append(dist)
    return listeClosest, listDistance


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(
        description='Calculate Inception v3 features for datasets'
    )

    
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=4, type=int, help='batch size')
    parser.add_argument('--folder', metavar='PATH', help='path to the folder picture file')
    parser.add_argument("--inspiration_method", type=str, default = "fullrandom", help = "Possible value is fullrandom/onlyinspiration") 
    parser.add_argument("--dataset_type", type = str, default = "unique", help = "Possible dataset type :unique/stellar")
    parser.add_argument("--multiview", action = "store_true")
    parser.add_argument('--labels', nargs='*', help='List of element used for classification', type=str, default = [])
    parser.add_argument('--labels_inspirationnal', nargs='*', help='List of element used for inspiration algorithm',type=str, default = [])
    parser.add_argument('--csv_path', type = str, default = None)   
    parser.add_argument('--generated_dataset', type = str, default = None)
    args = parser.parse_args()

    print("Start loading inception")
    cnn = create_cnn().to(device)
    cnn.normalization.mean =  cnn.normalization.mean.to(device)
    cnn.normalization.std = cnn.normalization.std.to(device)
    
    print("Data parallel")
    transform = transforms.Compose(
        [   
            transforms.Lambda(convert_transparent_to_rgb),
            transforms.Resize((args.size,args.size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    dataset = Dataset(args.folder,
        transform, args.size, 
        columns = args.labels,
        columns_inspirationnal = args.labels_inspirationnal,
        dataset_type = args.dataset_type,
        multiview = args.multiview,
        csv_path = args.csv_path
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        num_workers=4,
    )
    
    

    dset = SimpleDataset(args.generated_dataset, transform=transform, resolution=args.size)
    loader2 = DataLoader(dset, batch_size=args.batch, num_workers=4)
    features_test = extract_features(cnn, loader2, device).numpy()
    print(f'extracted {features_test.shape[0]} features')
    
    features = extract_features(cnn, loader, device).numpy()
    
    print(f'extracted {features.shape[0]} features')
    list_neighboors, list_distance = findNearestNeighboors(features,features_test)



    dic ={}
    for k in range(len(list_neighboors)) :
        dic[k] = []
        for i,index in enumerate(list_neighboors[k]):
            dic[k].append((dataset.df.iloc[index],list_distance[k][i]))

    print(dic)





