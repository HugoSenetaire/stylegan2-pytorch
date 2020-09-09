from .model import *
from .non_leaking import *

from torch import nn, autograd, optim



# NETWORK INITIALISATION :


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def calculate_reg_ratio(args):
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    return g_reg_ratio, d_reg_ratio


def create_network(args, dataset, device):

    latent_label_dim = dataset.get_len()

    generator = Generator(
        args.size, args.latent, args.n_mlp,
         channel_multiplier=args.channel_multiplier,
         latent_label_dim=latent_label_dim,
         mask = args.mask
    ).to(device)

    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier,
        latent_label_dim= latent_label_dim,
         dic_latent_label_dim=dataset.dic_column_dim,
         dic_inspirationnal_label_dim= dataset.dic_column_dim_inspirationnal,
         device=device,
         discriminator_type=args.discriminator_type
    ).to(device)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp,
         channel_multiplier=args.channel_multiplier,
         latent_label_dim=latent_label_dim,
         mask = args.mask,
    ).to(device)
   
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    return generator,discriminator,g_ema



def create_optimiser(args, generator,discriminator):
    g_reg_ratio, d_reg_ratio = calculate_reg_ratio(args)

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

    return g_optim, d_optim


def load_weights(args, generator, discriminator, g_ema, g_optim, d_optim):
    print("load model:", args.ckpt)

    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    g_reg_ratio, d_reg_ratio = calculate_reg_ratio(args)
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

