import argparse
import math
import random
import os
from nevergrad.optimization import optimizerlib
from copy import deepcopy

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm

from create_feature_extractor import *
from model import Generator, Discriminator
from dataset import MultiResolutionDataset, Dataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment

from create_feature_extractor import FeatureTransform

def getVal(kwargs, key, default):

    # out = kwargs.get(key, default)
    # if out is None:
    #     return default
    print(kwargs)
    print(key)
    print(default)
    out = kwargs[key]
    if out is None :
        out = default
    return out


def pil_loader(path):

    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def gradientDescentOnInput(model,
                           input,
                           featureExtractors,
                           imageTransforms,
                           dataset,
                           weights=None,
                           visualizer=None,
                           lambdaD=0.03,
                           nSteps=6000,
                           randomSearch=False,
                           nevergrad=None,
                           lr=1,
                           outPathSave=None):
    r"""
    Performs a similarity search with gradient descent.

    Args:

        model (BaseGAN): trained GAN model to use
        input (tensor): inspiration images for the gradient descent. It should
                        be a [NxCxWxH] tensor with N the number of image, C the
                        number of color channels (typically 3), W the image
                        width and H the image height
        featureExtractors (nn.module): list of networks used to extract features
                                       from an image
        weights (list of float): if not None, weight to give to each feature
                                 extractor in the loss criterion
        visualizer (visualizer): if not None, visualizer to use to plot
                                 intermediate results
        lambdaD (float): weight of the realism loss
        nSteps (int): number of steps to perform
        randomSearch (bool): if true, replace tha gradient descent by a random
                             search
        nevergrad (string): must be in None or in ['CMA', 'DE', 'PSO',
                            'TwoPointsDE', 'PortfolioDiscreteOnePlusOne',
                            'DiscreteOnePlusOne', 'OnePlusOne']
        outPathSave (string): if not None, path to save the intermediate
                              iterations of the gradient descent
    Returns

        output, optimalVector, optimalLoss

        output (tensor): output images
        optimalVector (tensor): latent vectors corresponding to the output
                                images
    """

    if nevergrad not in [None, 'CMA', 'DE', 'PSO',
                         'TwoPointsDE', 'PortfolioDiscreteOnePlusOne',
                         'DiscreteOnePlusOne', 'OnePlusOne']:
        raise ValueError("Invalid nevergard mode " + str(nevergrad))


    randomSearch = randomSearch or (nevergrad is not None)
    print("Running for %d setps" % nSteps)

    if visualizer is not None:
        visualizer.publishTensors(input, (128, 128))

    # Detect categories
    varNoise = torch.randn((input.size(0),
                            model.style_dim,
                            ),
                           requires_grad=True, device=model.device)

    optimNoise = optim.Adam([varNoise],
                            betas=[0., 0.99], lr=lr)

    #noiseOut = model.test(varNoise, getAvG=True, toCPU=False)
    label = dataset.random_one_hot(input.size(0))
    noiseOut = model.forward(varNoise,labels = label)

    if not isinstance(featureExtractors, list):
        featureExtractors = [featureExtractors]
    if not isinstance(imageTransforms, list):
        imageTransforms = [imageTransforms]

    nExtractors = len(featureExtractors)

    if weights is None:
        weights = [1.0 for i in range(nExtractors)]

    if len(imageTransforms) != nExtractors:
        raise ValueError(
            "The number of image transforms should match the number of \
            feature extractors")
    if len(weights) != nExtractors:
        raise ValueError(
            "The number of weights should match the number of feature\
             extractors")

    featuresIn = []
    for i in range(nExtractors):

        if len(featureExtractors[i]._modules) > 0:
            featureExtractors[i] = nn.DataParallel(
                featureExtractors[i]).train().to(model.device)

        featureExtractors[i].eval()
        imageTransforms[i] = nn.DataParallel(
            imageTransforms[i]).to(model.device)

        featuresIn.append(featureExtractors[i](
            imageTransforms[i](input.to(model.device))).detach())

        if nevergrad is None:
            featureExtractors[i].train()

    lr = 1

    optimalVector = None
    optimalLoss = None

    epochStep = int(nSteps / 3)
    gradientDecay = 0.1

    nImages = input.size(0)
    print(f"Generating {nImages} images")
    if nevergrad is not None:
        optimizers = []
        for i in range(nImages):
            # optimizers += [optimizerlib.registry[nevergrad](
            #     dimension=model.config.noiseVectorDim +
            #     model.config.categoryVectorDim,
            #     budget=nSteps)]
            optimizers += [optimizerlib.registry[nevergrad](
                dimension=model.style_dim,
                budget=nSteps)]

    def resetVar(newVal):
        newVal.requires_grad = True
        print("Updating the optimizer with learning rate : %f" % lr)
        varNoise = newVal
        optimNoise = optim.Adam([varNoise],
                                betas=[0., 0.99], lr=lr)

    # String's format for loss output
    formatCommand = ' '.join(['{:>4}' for x in range(nImages)])
    for iter in range(nSteps):

        optimNoise.zero_grad()
        model.netG.zero_grad()
        model.netD.zero_grad()

        if randomSearch:
            varNoise = torch.randn((nImages,
                                    model.style_dim),
                                   device=model.device)
            if nevergrad:
                inps = [] # inputs ?
                for i in range(nImages):
                    inps += [optimizers[i].ask()]
                    npinps = np.array(inps)

                varNoise = torch.tensor(
                    npinps, dtype=torch.float32, device=model.device)
                varNoise.requires_grad = True
                varNoise.to(model.device)

        noiseOut = model.forward(varNoise)
        sumLoss = torch.zeros(nImages, device=model.device)

        loss = (((varNoise**2).mean(dim=1) - 1)**2)
        sumLoss += loss.view(nImages)
        loss.sum(dim=0).backward(retain_graph=True)

        for i in range(nExtractors):
            featureOut = featureExtractors[i](imageTransforms[i](noiseOut))
            diff = ((featuresIn[i] - featureOut)**2)
            loss = weights[i] * diff.mean(dim=1)
            sumLoss += loss

            if not randomSearch:
                retainGraph = (lambdaD > 0) or (i != nExtractors - 1)
                loss.sum(dim=0).backward(retain_graph=retainGraph)

        if lambdaD > 0:

            loss = -lambdaD * model.netD(noiseOut)[:, 0]
            sumLoss += loss

            if not randomSearch:
                loss.sum(dim=0).backward()

        if nevergrad:
            for i in range(nImages):
                optimizers[i].tell(inps[i], float(sumLoss[i]))
        elif not randomSearch:
            optimNoise.step()

        if optimalLoss is None:
            optimalVector = deepcopy(varNoise)
            optimalLoss = sumLoss

        else:
            optimalVector = torch.where(sumLoss.view(-1, 1) < optimalLoss.view(-1, 1),
                                        varNoise, optimalVector).detach()
            optimalLoss = torch.where(sumLoss < optimalLoss,
                                      sumLoss, optimalLoss).detach()

        if iter % 100 == 0:
            if visualizer is not None:
                visualizer.publishTensors(noiseOut.cpu(), (128, 128))

                if outPathSave is not None:
                    index_str = str(int(iter/100))
                    outPath = os.path.join(outPathSave, index_str + ".jpg")
                    visualizer.saveTensor(
                        noiseOut.cpu(),
                        (noiseOut.size(2), noiseOut.size(3)),
                        outPath)

            print(str(iter) + " : " + formatCommand.format(
                *["{:10.6f}".format(sumLoss[i].item())
                  for i in range(nImages)]))

        if iter % epochStep == (epochStep - 1):
            lr *= gradientDecay
            resetVar(optimalVector)

    # output = model.test(optimalVector, getAvG=True, toCPU=True).detach()
    output = model.forward(optimalVector,labels = label).detach()

    if visualizer is not None:
        visualizer.publishTensors(
            output.cpu(), (output.size(2), output.size(3)))

    print("optimal losses : " + formatCommand.format(
        *["{:10.6f}".format(optimalLoss[i].item())
          for i in range(nImages)]))
    return output, optimalVector, optimalLoss



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
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment_p", type=float, default=0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--output_prefix", type=str, default = None)
    parser.add_argument("--iter", type=int, default = 2000)
    


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
                        help="Number of steps", default=6000)
    parser.add_argument('--weights', type=float, dest='weights',
                        nargs='*', help="Weight of each classifier. Default \
                        value is one. If specified, the number of weights must\
                        match the number of feature exatrcators.")
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


    
    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0
    size = args.size

    transform = transforms.Compose(
        [   
            transforms.Lambda(convert_transparent_to_rgb),
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    dataset = Dataset(args.path, transform, args.size)
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
         latent_label_dim=latent_label_dim
    ).to(device)
 
    g_ema = Generator(
        args.size, args.latent, args.n_mlp,
         channel_multiplier=args.channel_multiplier,
         latent_label_dim=latent_label_dim
    ).to(device)
    g_ema.eval()


    print(generator)
    print(dataset)

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        g_ema.load_state_dict(ckpt["g_ema"])


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


    # if get_rank() == 0 and wandb is not None and args.wandb:
    #     wandb.init(project="stylegan 2")

    # Get name for experience
    # name = getVal(args, "name", None)
    # name = args.name
    # if name is None:
        # raise ValueError("You need to input a name")

    



    # Get options
    # scale = getVal(args, "scale", None)
    # iter = getVal(args, "iter", None)
    # nRuns = getVal(args, "nRuns", 1)
    # scale = args.scale
    iter = args.iter
    nRuns = args.nRuns

    # Load the image
    ## targetSize = visualizer.model.getSize()
    targetSize = size
    ## baseTransform = standardTransform(targetSize)
    baseTransform = transform
    # visualizer = GANVisualizer(
    #     pathModel, modelConfig, modelType, visualisation)

    # Treating image
    # imgPath = getVal(args, "inputImage", None)
    imgPath = args.inputImage
    if imgPath is None:
        raise ValueError("You need to input an image path")
    img = pil_loader(imgPath)
    input = baseTransform(img)
    input = input.view(1, input.size(0), input.size(1), input.size(2))

    # pathsModel = getVal(kwargs, "featureExtractor", None)
    pathsModel = args.featureExtractor
    featureExtractors = []
    imgTransforms = []

    # if weights is not None:
    #     if pathsModel is None or len(pathsModel) != len(weights):
    #         raise AttributeError(
    #             "The number of weights must match the number of models")
    

    weights=None

    if pathsModel is not None:
        for path in pathsModel:
           
            featureExtractor, mean, std = buildFeatureExtractor(
                path, resetGrad=True)
            imgTransform = FeatureTransform(mean, std, size=args.size)
            featureExtractors.append(featureExtractor)
            imgTransforms.append(imgTransform)
    else :
        raise ValueError("Need to input a feature extractor model")


    basePath = os.path.join(args.output_prefix, args.suffix)

    if not os.path.isdir(basePath):
        os.mkdir(basePath)


    print("All results will be saved in " + basePath)

    outDictData = {}
    outPathDescent = None

    fullInputs = torch.cat([input for x in range(nRuns)], dim=0)

    if kwargs['save_descent']:
        outPathDescent = os.path.join(
            os.path.dirname(basePath), "descent")
        if not os.path.isdir(outPathDescent):
            os.mkdir(outPathDescent)

    img, outVectors, loss = gradientDescentOnInput(g_ema,
                                                   fullInputs,
                                                   featureExtractors,
                                                   imgTransforms,
                                                   dataset,
                                                   visualizer=visualisation,
                                                   lambdaD=kwargs['lambdaD'],
                                                   nSteps=kwargs['nSteps'],
                                                   weights=weights,
                                                   randomSearch=kwargs['random_search'],
                                                   nevergrad=kwargs['nevergrad'],
                                                   lr=kwargs['learningRate'],
                                                   outPathSave=outPathDescent)

    pathVectors = basePath + "vector.pt"
    torch.save(outVectors, open(pathVectors, 'wb'))

    path = basePath + ".jpg"
    visualisation.saveTensor(img, (img.size(2), img.size(3)), path)
    outDictData[os.path.splitext(os.path.basename(path))[0]] = \
        [x.item() for x in loss]

    outVectors = outVectors.view(outVectors.size(0), -1)
    outVectors *= torch.rsqrt((outVectors**2).mean(dim=1, keepdim=True))

    barycenter = outVectors.mean(dim=0)
    barycenter *= torch.rsqrt((barycenter**2).mean())
    meanAngles = (outVectors * barycenter).mean(dim=1)
    meanDist = torch.sqrt(((barycenter-outVectors)**2).mean(dim=1)).mean(dim=0)
    outDictData["Barycenter"] = {"meanDist": meanDist.item(),
                                 "stdAngles": meanAngles.std().item(),
                                 "meanAngles": meanAngles.mean().item()}

    path = basePath + "_data.json"
    outDictData["kwargs"] = kwargs

    with open(path, 'w') as file:
        json.dump(outDictData, file, indent=2)

    pathVectors = basePath + "vectors.pt"
    torch.save(outVectors, open(pathVectors, 'wb'))