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
import json
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

from utils import *
from loss import *
from non_leaking import augment
from parser_utils import *


from create_feature_extractor import FeatureTransform

def getVal(kwargs, key, default):
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

def buildFeatureExtractor(pathModel, resetGrad=True):

    modelData = torch.load(pathModel)

    fullDump = modelData.get("fullDump", False)
    if fullDump:
        model = modelData['model']
    else:
        modelType = loadmodule(
            modelData['package'], modelData['network'], prefix='')
        model = modelType(**modelData['kwargs'])
        model = cutModelHead(model)
        model.load_state_dict(modelData['data'])

    for param in model.parameters():
        param.requires_grad = resetGrad

    mean = modelData['mean']
    std = modelData['std']

    return model, mean, std


def gradientDescentOnInput(model,
                           discriminator,
                           input,
                           featureExtractors,
                           imageTransforms,
                           dataset,
                           device,
                           weights=None,
                           visualizer=None,
                           lambdaD=0.03,
                           nSteps=1000, #6000
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

   
    varNoise = [torch.randn(input.size(0),
                            model.style_dim,
                           requires_grad=True, device=device)]

    optimNoise = optim.Adam(varNoise,
                            betas=[0., 0.99], lr=lr)

    #noiseOut = model.test(varNoise, getAvG=True, toCPU=False)
    label = dataset.random_one_hot(input.size(0)).to(device)
    noiseOut = model(varNoise,labels = label)

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
                featureExtractors[i]).train().to(device)

        featureExtractors[i].eval()
        imageTransforms[i] = nn.DataParallel(
            imageTransforms[i]).to(device)

        featuresIn.append(featureExtractors[i](
            imageTransforms[i](input.to(device))).detach())

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
            # optimizers += [optimizerlib.registry[nevergrad](
            #     dimension=model.style_dim,
            #     budget=nSteps)]
            optimizers += [optimizerlib.registry[nevergrad](
                parametrization = model.style_dim,
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
        model.zero_grad()
        discriminator.zero_grad()

        if randomSearch:
            varNoise = [torch.randn((nImages,
                                    model.style_dim),
                                   device=device)]
            if nevergrad:
                inps = [] # inputs ?
                for i in range(nImages):
                    inps += [optimizers[i].ask()]
                    npinps = np.array(inps)
                
                # print(type(npinps),type(npinps[0]),type(npinps))
                # print(inps[0].args[0])

                varNoise = [torch.tensor(
                    inps[0].args[0], dtype=torch.float32, device=device)[None,:]]
                varNoise[0].requires_grad = True
                varNoise[0].to(device)

        # print(len(varNoise),varNoise[0].shape)
        noiseOut,_ = model(varNoise,labels = label)
        sumLoss = torch.zeros(nImages, device=device)
        # print(noiseOut.shape)
        loss = (((varNoise[0]**2).mean(dim=1) - 1)**2)
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

            loss = -lambdaD * discriminator(noiseOut,labels = label)[:, 0]
            sumLoss += loss

            if not randomSearch:
                loss.sum(dim=0).backward()

        if nevergrad:
            for i in range(nImages):
                optimizers[i].tell(inps[i], float(sumLoss[i]))
        elif not randomSearch:
            optimNoise.step()

        if optimalLoss is None:
            optimalVector = deepcopy(varNoise[0])
            optimalLoss = sumLoss

        else:
            optimalVector = torch.where(sumLoss.view(-1, 1) < optimalLoss.view(-1, 1),
                                        varNoise[0], optimalVector).detach()
            optimalLoss = torch.where(sumLoss < optimalLoss,
                                      sumLoss, optimalLoss).detach()

        if iter % 100 == 0:
            index_str = str(int(iter/100))
            utils.save_image(
                    noiseOut.detach(),
                    os.path.join(outPathSave, index_str + ".jpg"),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
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

    output,_ = model([optimalVector],labels = label)
    output = output.detach()

    if visualizer is not None:
        visualizer.publishTensors(
            output.cpu(), (output.size(2), output.size(3)))

    print("optimal losses : " + formatCommand.format(
        *["{:10.6f}".format(optimalLoss[i].item())
          for i in range(nImages)]))
    return output, optimalVector, optimalLoss



if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser_network.create_parser_network()
    parser_dataset.create_parser_dataset()
    parser_inspiration.create_parser_inspiration()
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
                transforms.RandomHorizontalFlip(),
                transforms.Resize((args.size,args.size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
    dataset = Dataset(args.folder,
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
        latent_label_dim= latent_label_dim,
        dic_latent_label_dim=dataset.dic_column_dim,
        dic_inspirationnal_label_dim= dataset.dic_column_dim_inspirationnal,
        device=device,
        discriminator_type=args.discriminator_type
    ).to(device)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp,
        channel_multiplier=args.channel_multiplier,
        latent_label_dim=latent_label_dim
    ).to(device)



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
        discriminator.load_state_dict(ckpt["d"])


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


    # Get option
    iter = args.iter
    nRuns = args.nRuns

    # Load the image
    targetSize = size
    baseTransform = transform

    # Treating image
    imgPath = args.inputImage
    if imgPath is None:
        raise ValueError("You need to input an image path")
    img = pil_loader(imgPath)
    input = baseTransform(img)
    input = input.view(1, input.size(0), input.size(1), input.size(2))

    pathsModel = args.featureExtractor
    featureExtractors = []
    imgTransforms = []


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
    outPathDescent = basePath

    fullInputs = torch.cat([input for x in range(nRuns)], dim=0)

    if args.save_descent:
        outPathDescent = os.path.join(
            os.path.dirname(basePath), "descent")
        if not os.path.isdir(outPathDescent):
            os.mkdir(outPathDescent)

    img, outVectors, loss = gradientDescentOnInput(g_ema,
                                                   discriminator,
                                                   fullInputs,
                                                   featureExtractors,
                                                   imgTransforms,
                                                   dataset,
                                                   device,
                                                   lambdaD=args.lambdaD,
                                                   nSteps=args.nSteps,
                                                   weights=weights,
                                                   randomSearch=args.random_search,
                                                   nevergrad=args.nevergrad,
                                                   lr=args.learningRate,
                                                   outPathSave=outPathDescent)

    pathVectors = basePath + "vector.pt"
    torch.save(outVectors, open(pathVectors, 'wb'))

    path = basePath + ".jpg"
    # torch.save(img, open(path,'wb'))

    utils.save_image(
                        img,
                        path,
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )
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
    outDictData["args"] = args

    with open(path, 'w') as file:
        json.dump(outDictData, file, indent=2)

    pathVectors = basePath + "vectors.pt"
    torch.save(outVectors, open(pathVectors, 'wb'))