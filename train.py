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
        progressive_manager(i,args, dataset, generator, discriminator, g_ema, g_optim, d_optim, device, sample_mask)
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        train_discriminator(args, generator, discriminator, dataset, loader, device, noise)
        

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_augment += torch.tensor(
                (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
            )
            ada_augment = reduce_sum(ada_augment)

            if ada_augment[1] > 255:
                pred_signs, n_pred = ada_augment.tolist()

                r_t_stat = pred_signs / n_pred

                if r_t_stat > args.ada_target:
                    sign = 1

                else:
                    sign = -1

                ada_aug_p += sign * ada_aug_step * n_pred
                ada_aug_p = min(1, max(0, ada_aug_p))
                ada_augment.mul_(0)

        d_regularize = i % args.d_reg_every == 0

        if d_regularize :
            real_img.requires_grad = True
            real_pred, real_classification, real_inspiration = discriminator(real_img,labels = real_label)
            if args.discriminator_type == 'AMGAN':
                real_pred = real_pred[:,len(dataset.columns)]
            else :
                if args.discriminator_type == 'bilinear':
                    real_pred = select_index_discriminator(real_pred, real_label)
                r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)


        random_label, random_dic_label, random_dic_inspiration = dataset.sample_manager(args.batch, device, "random", args.inspiration_method)
        if args.mask :
            random_mask = dataset.random_mask(args.batch)
            random_mask = random_mask.to(device)
        else :
            random_mask = None
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise,labels = random_label, mask = random_mask)
        if args.mask and args.mask_enforcer == "zero_based":
            zero_noise = mixing_noise(args.batch, args.latent, args.mixing, device, zero = True)
            zero_img, _ = generator(zero_noise, labels= random_label, mask = random_mask, noise = 'zero', randomize_noise = False)



        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        
        fake_pred, fake_classification, fake_inspiration = discriminator(fake_img,labels = random_label)
        if args.discriminator_type == "bilinear":
            fake_pred = select_index_discriminator(fake_pred,random_label)
        if args.discriminator_type == "AMGAN":
            # random_label = add_zero(random_label,device)
            # g_loss = classification_loss(fake_pred,random_label)
            g_loss = classification_loss(fake_pred, random_dic_label[dataset.columns[0]])
        else :
            g_loss = g_nonsaturating_loss(fake_pred)


        if args.mask :
            if args.mask_enforcer == "zero_based":
                shape_loss = g_shape_loss(zero_img, random_mask)

            elif args.mask_enforcer == "saturation":
                fake_img_grey_scale = convert_to_greyscale(fake_img)
                fake_img_grey_scale = normalisation(fake_img_grey_scale,normalisationLayer)
                random_mask_saturated = saturation(random_mask, device).squeeze()
                new_shape = saturation(fake_img_grey_scale, device)
                shape_loss = g_shape_loss(new_shape, random_mask_saturated)
            loss_dict["gclassic"] = g_loss
            g_loss += shape_loss
            loss_dict["gmask"] = shape_loss


        if args.discriminator_type == "design":
            if dataset.get_len()>0 :
                for column in dataset.columns :
                    g_loss += args.lambda_classif_gen * classification_loss(fake_classification[column], random_dic_label[column])
                for column in dataset.columns_inspirationnal :
                    g_loss += args.lambda_inspiration_gen * creativity_loss(fake_inspiration[column], random_dic_inspiration[column], device)
            


        loss_dict["g"] = g_loss
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)

            random_label, random_dic_label, random_dic_inspiration = dataset.sample_manager(path_batch_size, device, "random", args.inspiration_method)

            if args.mask :
                random_mask = dataset.random_mask(path_batch_size)
                random_mask = random_mask.to(device)

            else :
                random_mask = None

            fake_img, latents = generator(noise, labels = random_label, return_latents=True, mask = random_mask)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

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
