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
from loss import *


def generate(args, g_ema, device, mean_latent, dataset):

    with torch.no_grad():
        g_ema.eval()

        for i in tqdm(range(args.pics)):
           sample_z = torch.randn(args.n_sample, args.latent, device=device)
           sample_label, sample_dic_label, sample_dic_inspiration = dataset.sample_manager(args.n_sample, device, args.label_method, args.inspiration_method)

           sample, _ = g_ema([sample_z],labels = sample_label, truncation=args.truncation, truncation_latent=mean_latent)
           
           utils.save_image(
            sample,
            args.output_prefix + f'sample/{str(i).zfill(6)}.png',
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

if __name__ == '__main__':
    device = "cuda"
    parser = argparse.ArgumentParser()
    # Dataset parameters 
    parser.add_argument("path", type=str)
    parser.add_argument("--dataset_type", type = str, default = "unique", help = "Possible dataset type :unique/stellar")
    parser.add_argument("--multiview", action = "store_true")
    parser.add_argument("--labels", nargs='*', help='List of element used for classification', type=str, default = [])
    parser.add_argument("--labels_inspirationnal", nargs='*', help='List of element used for inspiration algorithm',type=str, default = [])
    parser.add_argument("--csv_path", type = str, default = None)  
    parser.add_argument("--inspiration_method", type=str, default = "fullrandom", help = "Possible value is fullrandom/onlyinspiration") 
    parser.add_argument("--label_method", type=str, default = "listing", help = "Possible value is random/listing")  


    # Network parameters
    parser.add_argument("--ckpt", type=str, default=None)


    # Training parameters
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)


    # Utils parameters :
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--output_prefix", type=str, default = None)
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--pics", type=int, default = 10)
    

    args = parser.parse_args()

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

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint['g_ema'])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent, dataset)
