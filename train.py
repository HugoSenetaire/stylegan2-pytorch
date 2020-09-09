import argparse
import math
import random
import os
import numpy as np
from tqdm import tqdm


import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
import torchvision

from utils import *
from loss import *
from network import *
from dataset import *
from parser import *
from parser_utils import *


try:
    import wandb

except ImportError:
    wandb = None




def train(args, loader, dataset, generator, discriminator, g_optim, d_optim, g_ema, device):

    # Iinitialisation :
    loader = sample_data(loader)
    pbar = range(args.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    ada_aug_step = args.ada_target / args.ada_length
    r_t_stat = 0


    # Sample Noise, Labels and Mask for feedback
    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    sample_label, sample_dic_label, sample_dic_inspiration = dataset.sample_manager(args.n_sample, device, args.label_method, args.inspiration_method)
    
    print("The labels for the generation are the following :")
    print(sample_dic_label)
    print("The weights for the generation are the following :")
    print(sample_dic_inspiration)

    sample_mask = sample_random_mask(args, args.n_sample, dataset, device, init=True, save_image= True)


    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break
        progressive_manager(i, args, dataset, generator, discriminator, g_ema, g_optim, d_optim, device, sample_mask)
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        d_loss = train_discriminator(i, args, generator, discriminator, dataset, loader, device, loss_dict, d_optim)
        

     


        requires_grad(generator, True)
        requires_grad(discriminator, False)

        g_loss = train_generator(i, args, generator, discriminator, dataset, loader, device, loss_dict, g_optim)

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()


        if args.mask :
            g_classic_loss = loss_reduced["gclassic"].mean().item()
            g_mask_loss = loss_reduced["gmask"].mean().item()


        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f};"
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f};"
                    f"augment: {ada_aug_p:.4f};"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % args.save_img_every == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z],labels = sample_label, mask = sample_mask)
                    utils.save_image(
                        sample,
                        os.path.join(args.output_prefix, f"sample/{str(i).zfill(6)}.png"),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % args.save_model_every == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                    },
                    os.path.join(args.output_prefix,f"checkpoint/{str(i).zfill(6)}.pt"),
                )





if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser_dataset.create_parser_dataset(parser)
    parser_network.create_parser_network(parser)
    parser_train.create_parser_train(parser)


    args = parser.parse_args()


    gpu_config(args)

    if not os.path.exists(os.path.join(args.output_prefix, "sample")):
        os.makedirs(os.path.join(args.output_prefix, "sample"))
    if not os.path.exists(os.path.join(args.output_prefix, "checkpoint")):
        os.makedirs(os.path.join(args.output_prefix, "checkpoint"))

 
    args.start_iter = 0

    dataset = create_dataset(args)
    
    loader = create_loader(args, dataset)

    generator, discriminator, g_ema =  create_network(args, dataset, device)
    g_optim,d_optim = create_optimiser(args, generator, discriminator)
   

    if args.ckpt is not None:
       load_weights(args,generator,discriminator,g_ema,g_optim,d_optim)
    if args.distributed:
        create_network_distributed(args, generator, discriminator)
    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    train(args, loader, dataset, generator, discriminator, g_optim, d_optim, g_ema, device)
