import argparse
import math
import random
import os

def create_parser_network(parser):
    parser.add_argument("--discriminator_type", type=str, default = "design", help = "option : bilinear/design ")
    parser.add_argument("--latent", type = int, default = 512)
    parser.add_argument("--n_mlp", type = int, default = 8)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--mask", action="store true")
    