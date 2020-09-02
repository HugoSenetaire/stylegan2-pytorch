import argparse
import math
import random
import os

def create_parser_train(parser):
    
    # Training parameters
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--progressive", action="store_true")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--upscale_every", type = int, default = 2000)
    parser.add_argument("--max_size",type=int, default = 512)
    parser.add_argument("--upscale_factor", type=int, default = 2)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment_p", type=float, default=0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--lambda_classif_gen", type=float, default = 1.0)
    parser.add_argument("--lambda_inspiration_gen", type=float, default=1.0)


    # Utils parameters :
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--output_prefix", type=str, default = None)
    parser.add_argument("--save_img_every", type=int, default = 100)
    parser.add_argument("--save_model_every", type = int, default = 1000)
    parser.add_argument("--n_sample", type=int, default=64)