import argparse
import math
import random
import os

def create_parser_train(parser):
    
 
    # Utils parameters :
    parser.add_argument("--wandb", action="store_true", help = "WanDB log")
    parser.add_argument("--local_rank", type=int, default=0, help = "Rank of the gpus used for training : Should not be given when")
    parser.add_argument("--output_prefix", type=str, default = None)
    parser.add_argument("--n_sample", type=int, default=64, help = "Number of sample generated for evaluation during training")
    parser.add_argument("--pics", type=int, default = 10)
    parser.add_argument("--truncation", type=float, default = 1.0)
    parser.add_argument("--truncation_mean", type=int, default = 4096)