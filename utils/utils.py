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


def create_label(batch_size,column_size,device):
    # TODO
    # Non nécessaire de le créer à chaque fois , juste mettre en global
    labels = torch.tensor([i%column_size for i in range(batch_size*column_size)]).to(device)
    return labels



def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises



def make_zero_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.zeros(batch, latent_dim, device=device)

    noises = torch.zeros(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises



def mixing_noise(batch, latent_dim, prob, device, zero = False):
    if not zero :
        if prob > 0 and random.random() < prob:
            return make_noise(batch, latent_dim, 2, device)

        else:
            return [make_noise(batch, latent_dim, 1, device)]
    else :
        if prob > 0 and random.random() < prob:
            return make_zero_noise(batch, latent_dim, 2, device)

        else:
            return [make_zero_noise(batch, latent_dim, 1, device)]

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

def add_scale(dataset,generator,discriminator,g_ema,g_optim,d_optim,device):
    generator.add_scale(g_optim,device =device)
    discriminator.add_scale(d_optim,device =device)

    g_ema.add_scale(device = device)

    dataset.upscale()






def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag



def sample_data(loader):
    while True:
        for batch in loader:
            yield batch