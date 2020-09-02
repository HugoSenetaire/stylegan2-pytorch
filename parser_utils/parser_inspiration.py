import argparse
import math
import random
import os

def create_parser_inspiration(parser):
    parser.add_argument("--output_prefix", type=str, default = None)
    parser.add_argument('-f', '--featureExtractor', help="Path to the feature \
                        extractor", nargs='*',
                        type=str, dest="featureExtractor")
    parser.add_argument('--input_image', type=str, dest="inputImage",
                        help="Path to the input image.")
    parser.add_argument('-N', type=int, dest="nRuns",
                        help="Number of gradient descent to run at the same \
                        time. Being too greedy may result in memory error.",
                        default=1)
    parser.add_argument('-l', type=float, dest="learningRate",
                        help="Learning rate",
                        default=1)
    parser.add_argument('-S', '--suffix', type=str, dest='suffix',
                        help="Output's suffix", default="inspiration")
    parser.add_argument('-R', '--rLoss', type=float, dest='lambdaD',
                        help="Realism penalty", default=0.03)
    parser.add_argument('--nSteps', type=int, dest='nSteps',
                        help="Number of steps", default=1000)
    parser.add_argument('--weights', type=float, dest='weights',
                        nargs='*', help="Weight of each classifier. Default \
                        value is one. If specified, the number of weights must\
                        match the number of feature extractors.")
    parser.add_argument('--gradient_descent', help='gradient descent',
                        action='store_true')
    parser.add_argument('--random_search', help='Random search',
                        action='store_true')
    parser.add_argument('--nevergrad', type=str,
                        choices=['CMA', 'DE', 'PSO', 'TwoPointsDE',
                                 'PortfolioDiscreteOnePlusOne',
                                 'DiscreteOnePlusOne', 'OnePlusOne'])
    parser.add_argument('--save_descent', help='Save descent',
                        action='store_true')