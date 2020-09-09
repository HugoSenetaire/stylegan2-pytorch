import argparse
import math
import random
import os

def create_parser_train(parser):
    
    # Training parameters
    parser.add_argument("--iter", type=int, default=800000, help = "int :Nb training iterations")
    parser.add_argument("--batch", type=int, default=16, help = "int :Number of element per iterations")
    parser.add_argument("--progressive", action="store_true", help = "bool :Progressive growing of network (ie double the scale of the current output)")
    parser.add_argument("--upscale_every", type = int, default = 2000, help = "int: Number of iterations separating a growth of the network")
    parser.add_argument("--max_size",type=int, default = 512, help = "int: Max size of the network and output")
    parser.add_argument("--upscale_factor", type=int, default = 2, help = "int: Factor for the upscale every parameter. If upscale factor is 2, first growth take args.upscale_every, second growth take args.upscale_every *2 and so on")
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16, help = "Regularization of discriminator every")
    parser.add_argument("--g_reg_every", type=int, default=4, help = "Regularization of generator every")
    parser.add_argument("--mixing", type=float, default=0.9, help = "Probability of mixing two styles during training")
    parser.add_argument("--lr", type=float, default=0.002, help = "Learning Rate")
    parser.add_argument("--augment", action="store_true", help = "Augmentation")
    parser.add_argument("--augment_p", type=float, default=0,help = "Augment probability")
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--lambda_classif_gen", type=float, default = 1.0, help = "Control the influence of the classification loss")
    parser.add_argument("--lambda_inspiration_gen", type=float, default=1.0, help = "Control the influence of the inspiration loss")
    parser.add_argument("--mask_enforcer", type=str, default = "zero_based",help = "Options : zero_based, saturation")


    # Utils parameters :
    parser.add_argument("--wandb", action="store_true", help = "WanDB log")
    parser.add_argument("--local_rank", type=int, default=0, help = "Rank of the gpus used for training : Should not be given when")
    parser.add_argument("--output_prefix", type=str, default = None)
    parser.add_argument("--save_img_every", type=int, default = 100)
    parser.add_argument("--save_model_every", type = int, default = 1000)
    parser.add_argument("--n_sample", type=int, default=64, help = "Number of sample generated for evaluation during training")