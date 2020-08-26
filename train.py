import argparse
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

try:
    import wandb

except ImportError:
    wandb = None

from model import Generator, Discriminator
from dataset import Dataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def classification_loss(pred, label):
    loss = nn.CrossEntropyLoss()
    return loss(pred,label)

def create_label(batch_size,column_size,device):
    # TODO
    # Non nécessaire de le créer à chaque fois , juste mettre en global
    labels = torch.tensor([i%column_size for i in range(batch_size*column_size)]).to(device)
    return labels



def creativity_loss(pred,weights,device):
    batch,column_size = weights.shape
    pred_aux = pred.unsqueeze(1)
    pred_aux = pred_aux.expand(-1,column_size,-1)
    pred_aux = pred_aux.view(batch*column_size,-1)
    neo_labels = create_label(batch,column_size,device)
    weights_aux = weights.flatten()
    pred_output = torch.nn.functional.cross_entropy(pred_aux,neo_labels, reduce=False)
    return torch.dot(weights_aux,pred_output)

    
def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


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

def add_scale(dataset,generator,discriminator,g_ema,g_optim,d_optim,device):
    generator.add_scale(g_optim,device =device)
    discriminator.add_scale(d_optim,device =device)

    g_ema.add_scale(device = device)
    dataset.image_size = dataset.image_size*2
    transform = transforms.Compose(
        [   
            transforms.Lambda(convert_transparent_to_rgb),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((dataset.image_size,dataset.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    dataset.transform = transform

    

def train(args, loader, dataset, generator, discriminator, g_optim, d_optim, g_ema, device):
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
    print(args.distributed)
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

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    sample_label, sample_dic_label, sample_dic_inspiration = dataset.sample_manager(args.n_sample, device, args.label_method, args.inspiration_method)
    
    print("The labels for the generation are the following :")
    print(sample_dic_label)
    print("The weights for the generation are the following :")
    print(sample_dic_inspiration)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        if args.progressive and i>0 :
            if i%args.upscale_every == 0 and dataset.image_size<args.max_size:
                print(f"Upscale at {i}")
                print(f"next upscale at {args.upscale_every*args.upscale_factor}")
                args.upscale_every = args.upscale_every*args.upscale_factor
                add_scale(dataset,generator,discriminator,g_ema,g_optim,d_optim,device)
                print(f"New size is {dataset.image_size}")

        real_label, real_img, real_dic_label, real_inspiration_label = next(loader)
        real_label = real_label.to(device)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        random_label, random_dic_label, random_dic_inspiration = dataset.sample_manager(args.batch, device, "random", args.inspiration_method)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
  
        fake_img, _ = generator(noise,labels= random_label)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred, fake_classification, fake_inspiration = discriminator(fake_img,labels = random_label) 
        real_pred, real_classification, real_inspiration = discriminator(real_img_aug, labels = real_label)
        if args.discriminator_type == "design":
            d_loss = d_logistic_loss(real_pred, fake_pred)
            if latent_label_dim>0 :
                for column in dataset.columns :
                    d_loss += classification_loss(real_classification[column], real_dic_label[column].to(device))
                for column in dataset.columns_inspirationnal :
                    d_loss += classification_loss(real_inspiration[column], real_inspiration_label[column].to(device))
        elif args.discriminator_type == "bilinear" :
            fake_pred = select_index_discriminator(fake_pred,random_label)
            print(real_label)
            print(real_pred)
            print(fake_pred)
            d_loss = d_logistic_loss(real_pred, fake_pred)

        
        

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

        if d_regularize:
            real_img.requires_grad = True
            real_pred, real_classification, real_inspiration = discriminator(real_img,labels = real_label)
            r1_loss = d_r1_loss(real_pred, real_img)
            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)
        

        random_label, random_dic_label, random_dic_inspiration = dataset.sample_manager(args.batch, device, "random", args.inspiration_method)
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise,labels = random_label)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        
        fake_pred, fake_classification, fake_inspiration = discriminator(fake_img,labels = random_label)
        if args.discriminator_type == "bilinear":
            fake_pred = select_index_discriminator(fake_pred,random_label)
        g_loss = g_nonsaturating_loss(fake_pred)

        # Not sure it is really necessary if no conditionning, would be for the creativity loss :
        if args.discriminator_type == "design":
            if latent_label_dim>0 :
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

            fake_img, latents = generator(noise, labels = random_label, return_latents=True)

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
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
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

            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z],labels = sample_label)
                    utils.save_image(
                        sample,
                        os.path.join(args.output_prefix, f"sample/{str(i).zfill(6)}.png"),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 1000 == 0:
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



def convert_rgb_to_transparent(image):
    if image.mode == 'RGB':
        return image.convert('RGBA')
    return image

def convert_transparent_to_rgb(image):
    if image.mode == 'RGBA':
        return image.convert('RGB')
    return image

if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str)
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--n_sample", type=int, default=64)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment_p", type=float, default=0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--output_prefix", type=str, default = None)
    parser.add_argument("--discriminator_type", type=str, default = "design", help = "option : bilinear/design ")
    parser.add_argument("--inspiration_method", type=str, default = "fullrandom", help = "Possible value is fullrandom/onlyinspiration") 
    parser.add_argument("--label_method", type=str, default = "listing", help = "Possible value is random/listing")
    parser.add_argument("--lambda_classif_gen", type=float, default = 1.0)
    parser.add_argument("--lambda_inspiration_gen", type=float, default=1.0)
    parser.add_argument("--progressive", action="store_true")
    parser.add_argument("--upscale_every", type = int, default = 2000)
    parser.add_argument("--max_size",type=int, default = 512)
    parser.add_argument("--upscale_factor", type=int, default = 2)
    parser.add_argument("--dataset_type", type = str, default = "unique", help = "Possible dataset type :unique/stellar")
    parser.add_argument("--multiview", action = "store_true")
    parser.add_argument('--labels', nargs='*', help='List of element used for classification', type=str, default = [])
    parser.add_argument('--labels_inspirationnal', nargs='*', help='List of element used for inspiration algorithm',type=str, default = [])
    parser.add_argument('--csv_path', type = str, default = None)    
    parser.add_argument('--latent', type = int, default = 512)
    parser.add_argument('--n_mlp', type = int, default = 8)
    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if not os.path.exists(os.path.join(args.output_prefix, "sample")):
        os.makedirs(os.path.join(args.output_prefix, "sample"))
    if not os.path.exists(os.path.join(args.output_prefix, "checkpoint")):
        os.makedirs(os.path.join(args.output_prefix, "checkpoint"))


    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.start_iter = 0

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
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    latent_label_dim = dataset.get_len()

   
    generator = Generator(
        args.size, args.latent, args.n_mlp,
         channel_multiplier=args.channel_multiplier,
         latent_label_dim=latent_label_dim,
    ).to(device)

    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier,
         dic_latent_label_dim=dataset.dic_column_dim,
         dic_inspirationnal_label_dim= dataset.dic_column_dim_inspirationnal,
         device=device
    ).to(device)


    g_ema = Generator(
        args.size, args.latent, args.n_mlp,
         channel_multiplier=args.channel_multiplier,
         latent_label_dim=latent_label_dim
    ).to(device)
   
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        try :
            g_optim.load_state_dict(ckpt["g_optim"])
            d_optim.load_state_dict(ckpt["d_optim"])
        except :
            g_optim = optim.Adam(
            generator.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
            )
            d_optim = optim.Adam(
                discriminator.parameters(),
                lr=args.lr * d_reg_ratio,
                betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
            )

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )


    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    train(args, loader, dataset, generator, discriminator, g_optim, d_optim, g_ema, device)
