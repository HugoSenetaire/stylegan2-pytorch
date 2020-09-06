import argparse
import math
import random
import os


def create_parser_fid(parser):
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--feature_path', type=str, default=None, required=True, help = 'extracted features from the dataset')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--ckpt_FID", nargs='*', help='List of element used for inspiration algorithm',type=str, default = [])