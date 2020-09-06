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
from dataset import Dataset
from utils import *
from parser_utils import *

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

def generate_samples(
    generator, inception, truncation, truncation_latent, batch_size, n_sample, device, args, label_name =None, element = None
):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 512, device=device)
        if label_name is None or label_name == 'None':
            sample_label, sample_dic_label, sample_dic_inspiration = dataset.sample_manager(batch, device, "random", args.inspiration_method)
        else :
            label_list = find_corresponding_label(label_name, batch_size)
            sample_label = dataset.category_manager( batch_size, device, label_list = label_list)
        img, _ = generator([latent],sample_label, truncation=truncation, truncation_latent=truncation_latent)

        utils.save_image(
                        img,
                        os.path.join(f"outputgenerate/{element}/{label_name}/{str(batch).zfill(6)}.png"),
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )

 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser_dataset.create_parser_dataset(parser)
    parser_network.create_parser_network(parser)
    parser_fid.create_parser_fid(parser)

   
    args = parser.parse_args()

    print(args)
    device = 'cuda'
    torch.cuda.set_device(args.local_rank)
    args.batch_size = 1

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
    dataset = Dataset(args.folder,
        transform, args.size, 
        columns = args.labels,
        columns_inspirationnal = args.labels_inspirationnal,
        dataset_type = args.dataset_type,
        multiview = args.multiview,
        csv_path = args.csv_path
    )


    latent_label_dim = dataset.get_len()

   
    g = Generator(
        args.size, args.latent, args.n_mlp,
         channel_multiplier=args.channel_multiplier,
         latent_label_dim=latent_label_dim
    ).to(device)


    for element in args.ckpt_FID :
        name_element = element.split("/")[-1].replace(".pt","")
        for category in args.limit_category :
            if not os.path.exists(os.path.join("outputgenerate",name_element,category)):
                os.makedirs(os.path.join("outputgenerate",name_element,category))
            ckpt = torch.load(element)
            print("element", element)
            print(category)
            g.load_state_dict(ckpt['g_ema'])
            g.eval()
            generate_samples(
                g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device, args, label_name = category, element = name_element
            )
