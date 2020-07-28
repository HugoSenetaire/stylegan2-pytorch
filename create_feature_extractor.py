
import argparse


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..networks.constant_net import FeatureTransform



class FeatureTransform(nn.Module):
    r"""
    Concatenation of a resize tranform and a normalization
    """

    def __init__(self,
                 mean=None,
                 std=None,
                 size=224):

        super(FeatureTransform, self).__init__()
        self.size = size

        if mean is None:
            mean = [0., 0., 0.]

        if std is None:
            std = [1., 1., 1.]

        self.register_buffer('mean', torch.tensor(
            mean, dtype=torch.float).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(
            std, dtype=torch.float).view(1, 3, 1, 1))

        if size is None:
            self.upsamplingModule = None
        else:
            self.upsamplingModule = torch.nn.Upsample(
                (size, size), mode='bilinear')

    def forward(self, x):

        if self.upsamplingModule is not None:
            x = self.upsamplingModule(x)

        x = x - self.mean
        x = x / self.std

        return x


def extractRelUIndexes(sequence, layers):

    layers.sort()

    index = 0
    output = []

    indexRef = 0
    indexScale = 1

    hasCaughtRelUOnLayer = False
    while indexRef < len(layers) and index < len(sequence):

        if isinstance(sequence[index], torch.nn.ReLU):

            if not hasCaughtRelUOnLayer and indexScale == layers[indexRef]:

                hasCaughtRelUOnLayer = True
                output.append(index)
                indexRef += 1

        if isinstance(sequence[index], torch.nn.MaxPool2d) \
                or isinstance(sequence[index], torch.nn.AvgPool2d):

            hasCaughtRelUOnLayer = False
            indexScale += 1

        index += 1

    return output


def extractIndexedLayers(sequence,
                         x,
                         indexes,
                         detach):

    index = 0
    output = []

    indexes.sort()

    for iSeq, layer in enumerate(sequence):

        if index >= len(indexes):
            break

        x = layer(x)

        if iSeq == indexes[index]:
            if detach:
                output.append(x.view(x.size(0), x.size(1), -1).detach())
            else:
                output.append(x.view(x.size(0), x.size(1), -1))
            index += 1

    return output


class LossTexture(torch.nn.Module):
    r"""
    An implenetation of style transfer's (http://arxiv.org/abs/1703.06868) like
    loss.
    """

    def __init__(self,
                 device,
                 modelName,
                 scalesOut):
        r"""
        Args:
            - device (torch.device): torch.device("cpu") or
                                     torch.device("cuda:0")
            - modelName (string): name of the torchvision.models model. For
                                  example vgg19
            - scalesOut (list): index of the scales to extract. In the Style
                                transfer paper it was [1,2,3,4]
        """

        super(LossTexture, self).__init__()
        scalesOut.sort()

        model = loadmodule("torchvision.models", modelName, prefix='')
        self.featuresSeq = model(pretrained=True).features.to(device)
        self.indexLayers = extractRelUIndexes(self.featuresSeq, scalesOut)

        self.reductionFactor = [1 / float(2**(i - 1)) for i in scalesOut]

        refMean = [2*p - 1 for p in[0.485, 0.456, 0.406]]
        refSTD = [2*p for p in [0.229, 0.224, 0.225]]

        self.imgTransform = FeatureTransform(mean=refMean,
                                             std=refSTD,
                                             size=None)

        self.imgTransform = self.imgTransform.to(device)

    def getLoss(self, fake, reals, mask=None):

        featuresReals = self.getFeatures(
            reals, detach=True, prepImg=True, mask=mask).mean(dim=0)
        featuresFakes = self.getFeatures(
            fake, detach=False, prepImg=True, mask=None).mean(dim=0)

        outLoss = ((featuresReals - featuresFakes)**2).mean()
        return outLoss

    def getFeatures(self, image, detach=True, prepImg=True, mask=None):

        if prepImg:
            image = self.imgTransform(image)

        fullSequence = extractIndexedLayers(self.featuresSeq,
                                            image,
                                            self.indexLayers,
                                            detach)
        outFeatures = []
        nFeatures = len(fullSequence)

        for i in range(nFeatures):

            if mask is not None:
                locMask = (1. + F.upsample(mask,
                                           size=(image.size(2) * self.reductionFactor[i],
                                                 image.size(3) * self.reductionFactor[i]),
                                           mode='bilinear')) * 0.5
                locMask = locMask.view(locMask.size(0), locMask.size(1), -1)

                totVal = locMask.sum(dim=2)

                meanReals = (fullSequence[i] * locMask).sum(dim=2) / totVal
                varReals = (
                    (fullSequence[i]*fullSequence[i] * locMask).sum(dim=2) / totVal) - meanReals*meanReals

            else:
                meanReals = fullSequence[i].mean(dim=2)
                varReals = (
                    (fullSequence[i]*fullSequence[i]).mean(dim=2))\
                     - meanReals*meanReals

            outFeatures.append(meanReals)
            outFeatures.append(varReals)

        return torch.cat(outFeatures, dim=1)

    def forward(self, x, mask=None):

        return self.getFeatures(x, detach=False, prepImg=False, mask=mask)

    def saveModel(self, pathOut):

        torch.save(dict(model=self, fullDump=True,
                        mean=self.imgTransform.mean.view(-1).tolist(),
                        std=self.imgTransform.std.view(-1).tolist()),
                   pathOut)


def loadmodule(package, name, prefix='..'):
    r"""
    A dirty hack to load a module from a string input

    Args:
        package (string): package name
        name (string): module name

    Returns:
        A pointer to the loaded module
    """
    strCmd = "from " + prefix + package + " import " + name + " as module"
    exec(strCmd)
    return eval('module')



def cutModelHead(model):

    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)

    return model


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



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('model_name', type=str,
                        choices=["vgg19", "vgg16"],
                        help="""Name of the desured featire extractor:
                        - vgg19, vgg16 : a variation of the style transfer \
                        feature developped in \
                        http://arxiv.org/abs/1703.06868""")
    parser.add_argument('--layers', type=int, nargs='*',
                        help="For vgg models only. Layers to select. \
                        Default ones are 3, 4, 5.", default=None)
    parser.add_argument('output_path', type=str,
                        help="""Path of the output feature extractor""")

    args = parser.parse_args()

    if args.model_name in ["vgg19", "vgg16"]:
        if args.layers is None:
            args.layers = [3, 4, 5]
        featureExtractor = LossTexture(torch.device("cpu"),
                                       args.model_name,
                                       args.layers)
        featureExtractor.saveModel(args.output_path)
    else:
        raise AttributeError(args.model_name + " not implemented yet")
