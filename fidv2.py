import argparse
import pickle

import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm

from model import Generator
from calc_inception import load_patched_inception_v3
from dataset import *
from torchvision import transforms, utils
from parser_utils import *
from utils import *

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
from dataset import Dataset, SimpleDataset
from utils import *
from parser_utils import *



@torch.no_grad()


def extract_features(loader, inception, device):
    pbar = tqdm(loader)

    feature_list = []

    for img in pbar:
        img = img.to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

    features = torch.cat(feature_list, 0)

    return features

def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser_dataset.create_parser_dataset(parser)
    parser_network.create_parser_network(parser)
    parser_fid.create_parser_fid(parser)
    parser.add_argument("--generated_images", type=str)  
    

   
    args = parser.parse_args()

    print(args)
    device = 'cuda'
    torch.cuda.set_device(args.local_rank)

    args.latent = 512
    args.n_mlp = 8
    

    transform = transforms.Compose(
        [   
            transforms.Lambda(convert_transparent_to_rgb),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((args.size,args.size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
  

   
    inception = load_patched_inception_v3().to(device)
    inception.eval()
    # inception = None
    for element in args.ckpt_FID :
        name_element = element.split("/")[-1].replace(".pt","")
        for category in args.limit_category :
            print(element)
            print(category)
            

            folder_total = os.path.join(os.path.join(args.generated_images,name_element,category))
            dataset = SimpleDataset(folder_total,
            transform, args.size
            )

            loader = DataLoader(
                dataset,
                batch_size=args.batch,
                num_workers=4,
            )
            features = extract_features(loader, inception, device).numpy()
            print(f'extracted {features.shape[0]} features')
            sample_mean = np.mean(features, 0)
            sample_cov = np.cov(features, rowvar=False)
            with open(args.feature_path.replace(".pkl",f"_{category}.pkl"), 'rb') as f:
                embeds = pickle.load(f)
                real_mean = embeds['mean']
                real_cov = embeds['cov']

            fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

            print('fid:', fid)
