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
from dataset import Dataset
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

def image_loader(image_name):
    image = Image.open(image_name)
    loader = transforms.Compose([
        transforms.Resize((512, 512)),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
#                                img,
#                                final_layer= ['pool_16']):
#     # cnn = copy.deepcopy(cnn)
#     # normalization module
#     normalization = Normalization(normalization_mean, normalization_std).to(device)
#     # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
#     # to put in modules that are supposed to be activated sequentially
#     model = nn.Sequential(normalization)
#     final = []
#     i = 0  # increment every time we see a conv
#     for layer in cnn.children():
#         if isinstance(layer, nn.Conv2d):
#             i += 1
#             name = 'conv_{}'.format(i)
#         elif isinstance(layer, nn.ReLU):
#             name = 'relu_{}'.format(i)
#             # The in-place version doesn't play very nicely with the ContentLoss
#             # and StyleLoss we insert below. So we replace with out-of-place
#             # ones here.
#             layer = nn.ReLU(inplace=False)
#         elif isinstance(layer, nn.MaxPool2d):
#             name = 'pool_{}'.format(i)
#         elif isinstance(layer, nn.BatchNorm2d):
#             name = 'bn_{}'.format(i)
#         else:
#             raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
#         model.add_module(name, layer)
#         if name in final_layer:
#             # add style loss:
#             with torch.no_grad():
#                 target_feature = model(img)
#                 final.append(torch.flatten(target_feature, start_dim=1, end_dim=-1))
#     return final[0]


def compute_image_embedding(directory_images, sku):
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    sku = directory_images+sku+'.jpg'
    img_list = [image_loader(sku)]
    final = get_style_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std, img_list)
    return final


def extract_features(cnn, loader, inception, device):
    pbar = tqdm(loader)

    feature_list = []

    for _,img,_,_ in pbar:
        img = img.to(device)
        feature = cnn(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

    features = torch.cat(feature_list, 0)

    return features


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(
        description='Calculate Inception v3 features for datasets'
    )

    
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=64, type=int, help='batch size')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--path', metavar='PATH', help='path to datset lmdb file')
    parser.add_argument("--inspiration_method", type=str, default = "fullrandom", help = "Possible value is fullrandom/onlyinspiration") 
    parser.add_argument("--dataset_type", type = str, default = "unique", help = "Possible dataset type :unique/stellar")
    parser.add_argument("--multiview", action = "store_true")
    parser.add_argument('--labels', nargs='*', help='List of element used for classification', type=str, default = [])
    parser.add_argument('--labels_inspirationnal', nargs='*', help='List of element used for inspiration algorithm',type=str, default = [])
    parser.add_argument('--csv_path', type = str, default = None)   
    parser.add_argument('--image_dataset', type = str, default = None)
    args = parser.parse_args()

    print("Start loading inception")
    inception = load_patched_inception_v3()
    print("Inception loaded")
    inception = nn.DataParallel(inception).eval().to(device)
    print("Data parallel")
    transform = transforms.Compose(
        [   
            transforms.Lambda(convert_transparent_to_rgb),
            transforms.Resize((args.size,args.size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    dataset = Dataset(args.path,
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
    # dset = MultiResolutionDataset(args.path, transform=transform, resolution=args.size)
    # loader = DataLoader(dset, batch_size=args.batch, num_workers=4)

    features = extract_features(loader, inception, device).numpy()

    features = features[: args.n_sample]

    print(f'extracted {features.shape[0]} features')

    mean = np.mean(features, 0)
    cov = np.cov(features, rowvar=False)

    name = os.path.splitext(os.path.basename(args.path))[0]

    with open(f'inception_{name}.pkl', 'wb') as f:
        pickle.dump({'mean': mean, 'cov': cov, 'size': args.size, 'path': args.path}, f)
