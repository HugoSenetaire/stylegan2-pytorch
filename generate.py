import argparse

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from tqdm import tqdm
from torchvision import utils, transforms
from model import Generator
from tqdm import tqdm
from dataset import Dataset



from utils import *
from torch.utils import data


def generate(args, g_ema, device, mean_latent, dataset):

    with torch.no_grad():
        g_ema.eval()

        for i in tqdm(range(args.pics)):
           sample_z = torch.randn(args.n_sample, args.latent, device=device)
           sample_label, sample_dic_label, sample_dic_inspiration = dataset.sample_manager(args.n_sample, device, args.label_method, args.inspiration_method)

           sample, _ = g_ema([sample_z],labels = sample_label, truncation=args.truncation, truncation_latent=mean_latent)
           
           
           utils.save_image(
            sample,
            os.path.join(args.output_prefix, f'sample_generate/{str(i).zfill(6)}.png'),
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

if __name__ == '__main__':
    device = "cuda"
    parser = argparse.ArgumentParser()
    # Dataset parameters 
    create_parser_dataset(parser)
    # Network parameters
    create_parser_network(parser)

    create_parser_generate(parser)


    

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8



    if not os.path.exists(os.path.join(args.output_prefix, "sample_generate")):
        os.makedirs(os.path.join(args.output_prefix, "sample_generate"))
    transform = transforms.Compose(
        [   
            transforms.Lambda(convert_transparent_to_rgb),
            transforms.RandomHorizontalFlip(),
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
    latent_label_dim = dataset.get_len()

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, latent_label_dim= latent_label_dim
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint['g_ema'])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent, dataset)
