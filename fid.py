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

@torch.no_grad()


def find_corresponding_label(label_name, batch_size):
    
    label_index = dataset.get_onehot_fromvalue(label_name).astype(int)
    label_list = np.ones((batch_size,1),dtype = int) * label_index
    return label_list



def extract_feature_from_samples(
    generator, inception, truncation, truncation_latent, batch_size, n_sample, device, args, label_name =None
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
            print(label_list)
            sample_label = dataset.category_manager( batch_size, device, label_list = label_list)
        img, _ = generator([latent],sample_label, truncation=truncation, truncation_latent=truncation_latent)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)

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

    # inception = load_patched_inception_v3().to(device)
    # inception.eval()
    inception = None
    for element in args.ckpt_FID :
        for category in args.limit_category :
            ckpt = torch.load(element)
            print("element", element)
            print(category)
            g.load_state_dict(ckpt['g_ema'])
            g.eval()

            if args.truncation < 1:
                with torch.no_grad():
                    mean_latent = g.mean_latent(args.truncation_mean)

            else:
                mean_latent = None


            features = extract_feature_from_samples(
                g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device, args, label_name = category
            ).numpy()
            print(f'extracted {features.shape[0]} features')

            sample_mean = np.mean(features, 0)
            sample_cov = np.cov(features, rowvar=False)

            with open(args.feature_path.replace(".pkl",f"_{category}.pkl"), 'rb') as f:
                embeds = pickle.load(f)
                real_mean = embeds['mean']
                real_cov = embeds['cov']

            fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

            print('fid:', fid)
