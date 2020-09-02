import argparse
import math
import random
import os


def create_parser_dataset(parser):
    parser.add_argument("--folder", type=str)
    parser.add_argument("--dataset_type", type = str, default = "unique", help = "Possible dataset type :unique/stellar")
    parser.add_argument("--multiview", action = "store_true")
    parser.add_argument("--labels", nargs='*', help='List of element used for classification', type=str, default = [])
    parser.add_argument("--labels_inspirationnal", nargs='*', help='List of element used for inspiration algorithm',type=str, default = [])
    parser.add_argument("--csv_path", type = str, default = None)  
    parser.add_argument("--inspiration_method", type=str, default = "fullrandom", help = "Possible value is fullrandom/onlyinspiration") 
    parser.add_argument("--label_method", type=str, default = "listing", help = "Possible value is random/listing")  

