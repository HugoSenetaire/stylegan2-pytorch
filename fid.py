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

def convert_rgb_to_transparent(image):
    if image.mode == 'RGB':
        return image.convert('RGBA')
    return image

def convert_transparent_to_rgb(image):
    if image.mode == 'RGBA':
        return image.convert('RGB')
    return image
@torch.no_grad()
def extract_feature_from_samples(
    generator, inception, truncation, truncation_latent, batch_size, n_sample, device, args
):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 512, device=device)
        sample_label, sample_dic_label, sample_dic_inspiration = dataset.sample_manager(batch, device, "random", args.inspiration_method)
        img, _ = g(latent,sample_label, truncation=truncation, truncation_latent=truncation_latent)
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
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--inception', type=str, default=None, required=True)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument('ckpt', metavar='CHECKPOINT')
    parser.add_argument("--inspiration_method", type=str, default = "fullrandom", help = "Possible value is fullrandom/onlyinspiration") 
    parser.add_argument("--dataset_type", type = str, default = "unique", help = "Possible dataset type :unique/stellar")
    parser.add_argument("--multiview", action = "store_true")
    parser.add_argument('--labels', nargs='*', help='List of element used for classification', type=str, default = [])
    parser.add_argument('--labels_inspirationnal', nargs='*', help='List of element used for inspiration algorithm',type=str, default = [])
    parser.add_argument('--csv_path', type = str, default = None)    
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)


    dataset = Dataset(args.path,
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

    g.load_state_dict(ckpt['g_ema'])
    g = nn.DataParallel(g)
    g.eval()

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g.mean_latent(args.truncation_mean)

    else:
        mean_latent = None

    inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    inception.eval()

    features = extract_feature_from_samples(
        g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device, args
    ).numpy()
    print(f'extracted {features.shape[0]} features')

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    with open(args.inception, 'rb') as f:
        embeds = pickle.load(f)
        real_mean = embeds['mean']
        real_cov = embeds['cov']

    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

    print('fid:', fid)
