import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm

from .loss import *
from .network import *
from .dataset import *
from .distributed import *



# Operation on tensors :
def convert_to_greyscale(tensor):
    tensor_greyscale = 0.21 * tensor[:,0,:,:] + 0.72 * tensor[:,1,:,:]  + 0.07 * tensor[:,2,:,:]
    return tensor_greyscale

def saturation(tensor,device):
    return torch.where(tensor<0.98 * tensor.max(), torch.ones(tensor.shape).to(device)*-1., torch.ones(tensor.shape).to(device)*1.)

def normalisation(tensor, normalisationLayer):
    tensor = normalisationLayer(tensor)
    return tensor

def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def select_index_discriminator(output_discriminator, label):
    if label == None and output_discriminator.shape[1]==1:
        return output_discriminator.squeeze(0)    
    index = torch.ge(label,0.5)
    filtered_output = output_discriminator.masked_select(index)
    return filtered_output


def add_zero(tensor,device):
    batch_size = tensor.shape[0]
    new_zero = torch.zeros((batch_size,1)).to(device)
    tensor = torch.cat([tensor,new_zero],dim =1).long()
    return tensor.to(device)

def create_fake_label_vector(tensor,device):
    batch_size = tensor.shape[0]
    column_size = tensor.shape[1]
    fake_label = torch.zeros((batch_size,column_size+1))
    fake_label[:,-1] = torch.ones((batch_size,)).to(device)
    return fake_label.long().to(device)

def create_fake_label(tensor,device):
    batch_size = tensor.shape[0]
    column_size = tensor.shape[1]
    fake_label = torch.ones((batch_size,),dtype=torch.long)*column_size
    return fake_label.to(device)



def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag



# Managing progressive transformation :

def add_scale(dataset,generator,discriminator,g_ema,g_optim,d_optim,device):
    generator.add_scale(g_optim,device =device)
    discriminator.add_scale(d_optim,device =device)
    g_ema.add_scale(device = device)
    dataset.upscale()

def progressive_manager(i,args, dataset, generator, discriminator, g_ema, g_optim, d_optim, device, sample_mask):
    if args.progressive and i>0 :
        if i%args.upscale_every == 0 and dataset.image_size<args.max_size:
            print(f"Upscale at {i}")
            print(f"next upscale at {args.upscale_every*args.upscale_factor}")
            args.upscale_every = args.upscale_every*args.upscale_factor
            add_scale(dataset,generator,discriminator,g_ema,g_optim,d_optim,device)
            if args.mask_enforcer == "saturation":
                normalisationLayer = nn.LayerNorm([dataset.image_size,dataset.image_size],elementwise_affine = False).to(device)
            print(f"New size is {dataset.image_size}")
            if args.mask :
                sample_mask = dataset.random_mask(args.n_sample).to(device)
                utils.save_image(
                                sample_mask,
                                os.path.join(args.output_prefix, f"sample_mask_{i}.png"),
                                nrow=int(args.n_sample ** 0.5),
                                normalize=True,
                                range=(-1, 1),
                            )




#SAMPLING UTILS :

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch



def sample_loader(loader, device):
    real_label, real_img, real_dic_label, real_inspiration_label, real_mask = next(loader)
    real_label = real_label.to(device)
    real_img = real_img.to(device)
    real_mask = real_mask.to(device)

    return real_label, real_img, real_dic_label, real_inspiration_label, real_mask


def sample_random(args, batch, dataset,device):
    random_label, random_dic_label, random_dic_inspiration = dataset.sample_manager(batch, device, "random", args.inspiration_method)
    random_mask= sample_random_mask(args, batch, dataset,device)

    return random_label, random_dic_label, random_dic_inspiration, random_mask 

def sample_random_mask(args, batch, dataset, device, init = False, save_image =False):
    if args.mask :
        random_mask = dataset.random_mask(batch).to(device)
        if save_image :
            utils.save_image(
                            random_mask,
                            os.path.join(args.output_prefix, f"sample_mask.png"),
                            nrow=int(batch** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
        if args.mask_enforcer == "saturation" and init:
            normalisationLayer = nn.LayerNorm([dataset.image_size,dataset.image_size], elementwise_affine = False).to(device)
    else :
        random_mask = None


    return random_mask


# FEEDBACK :

def save_image(i, args, g_ema, sample_z, sample_label, sample_mask):
    with torch.no_grad():
        g_ema.eval()
        sample, _ = g_ema([sample_z],labels = sample_label, mask = sample_mask)
        print(sample)
        utils.save_image(
            sample,
            os.path.join(args.output_prefix, f"sample/{str(i).zfill(6)}.png"),
            nrow=int(args.n_sample ** 0.5),
            normalize=True,
            range=(-1, 1),
        )

def save_weights(i, args, g_module, d_module, g_ema, g_optim, d_optim):
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

def save_checkpoint(i, args, g, d, g_ema, g_optim, d_optim, sample_z, sample_label, sample_mask):
    if i % args.save_img_every == 0:
        save_image(i, args, g_ema, sample_z, sample_label, sample_mask)
    if i % args.save_model_every == 0:
        save_weights(i, args, g, d, g_ema, g_optim, d_optim)
    

def get_total_feedback(loss_dict, args):
    loss_reduced = reduce_loss_dict(loss_dict)
    loss_reduced_feedback = {}
    loss_reduced_feedback["Discriminator"] = loss_reduced["d"].mean().item()
    loss_reduced_feedback["Generator"] = loss_reduced["g"].mean().item()


    if args.mask :
        loss_reduced_feedback["Generator Classic"] = loss_reduced["gclassic"].mean().item()
        loss_reduced_feedback["Generator mask Loss"] = loss_reduced["gmask"].mean().item()

    loss_reduced_feedback["R1"] = loss_reduced["r1"].mean().item()
    loss_reduced_feedback["Path Length Regularization"] = loss_reduced["path"].mean().item()
    loss_reduced_feedback["Real Score"]  = loss_reduced["real_score"].mean().item()
    loss_reduced_feedback["Fake Score"]  = loss_reduced["fake_score"].mean().item()
    loss_reduced_feedback["Path Length"] = loss_reduced["path_length"].mean().item()

    return loss_reduced_feedback

# NETWORK TRAINING :

def init_training(device):
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {"path": path_loss,
     "path_length": path_lengths,
     "mean_path_length_avg": mean_path_length_avg,
     }
    return d_loss_val, r1_loss, g_loss_val, path_loss, path_lengths, mean_path_length_avg, loss_dict




def train_discriminator(i, args, generator, discriminator, dataset, loader, device, loss_dict, d_optim):
    noise = dataset.mixing_noise(args.batch, args.latent, args.mixing, device)
    real_label, real_img, real_dic_label, real_inspiration_label, real_mask = sample_loader(loader,device)
    random_label, random_dic_label, random_dic_inspiration, random_mask = sample_random(args, args.batch, dataset, device)
    
    fake_img, _ = generator(noise,labels= random_label, mask = random_mask)
    if args.augment:
        real_img_aug, _ = augment(real_img, ada_aug_p)
        fake_img, _ = augment(fake_img, ada_aug_p)
    else:
        real_img_aug = real_img

    
    fake_pred, fake_classification, fake_inspiration = discriminator(fake_img,labels = random_label) 
    real_pred, real_classification, real_inspiration = discriminator(real_img, labels = real_label)
    

    if args.discriminator_type == "design":
        d_loss = d_logistic_loss(real_pred, fake_pred)
        if dataset.get_len()>0 :
            for column in dataset.columns :
                d_loss += classification_loss(real_classification[column], real_dic_label[column].to(device))
            for column in dataset.columns_inspirationnal :
                d_loss += classification_loss(real_inspiration[column], real_inspiration_label[column].to(device))
    elif args.discriminator_type == "bilinear" :
        fake_pred = select_index_discriminator(fake_pred,random_label)
        real_pred = select_index_discriminator(real_pred, real_label)
        d_loss = d_logistic_loss(real_pred, fake_pred)
    elif args.discriminator_type == "AMGAN":
        fake_label = create_fake_label(random_label,device)
        real_label = real_dic_label[dataset.columns[0]].to(device)
        d_loss = classification_loss(fake_pred,fake_label) + classification_loss(real_pred,real_label)

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

    return d_loss


def train_generator(i, args, generator, discriminator, dataset, loader, device, loss_dict, g_optim, mean_path_length, mean_path_length_avg) :
    
    random_label, random_dic_label, random_dic_inspiration = dataset.sample_manager(args.batch, device, "random", args.inspiration_method)
    if args.mask :
        random_mask = dataset.random_mask(args.batch)
        random_mask = random_mask.to(device)
    else :
        random_mask = None
    noise = dataset.mixing_noise(args.batch, args.latent, args.mixing, device)
    fake_img, _ = generator(noise,labels = random_label, mask = random_mask)
    if args.mask and args.mask_enforcer == "zero_based":
        zero_noise = dataset.mixing_noise(args.batch, args.latent, args.mixing, device, zero = True)
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
        noise = dataset.mixing_noise(path_batch_size, args.latent, args.mixing, device)
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
        loss_dict["mean_path_length_avg"]=mean_path_length_avg
        loss_dict["mean_path_length"] = mean_path_length
        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()




    return g_loss