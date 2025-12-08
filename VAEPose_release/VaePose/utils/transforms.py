import logging
import torch
import torchvision
import numpy as np

logger = logging.getLogger(__name__)


class Template:

    def __init__(self, a=0.01, b=0.05):
        self.a = a
        self.b = a

    def __call__(self, sample):

        return sample


class ToTensor():

    def __init__(self):
        pass

    def __call__(self, input):
        
        return input


def fetch_transforms(cfg):

    if cfg.data.transforms_type == "h2onet":
        train_transforms = [torchvision.transforms.ToTensor()]
        test_transforms = [torchvision.transforms.ToTensor()]

    else:
        raise NotImplementedError

    logger.info("Train transforms: {}".format(", ".join([type(t).__name__ for t in train_transforms])))
    logger.info("Val/Test transforms: {}".format(", ".join([type(t).__name__ for t in test_transforms])))
    train_transforms = torchvision.transforms.Compose(train_transforms)
    test_transforms = torchvision.transforms.Compose(test_transforms)
    return train_transforms, test_transforms
