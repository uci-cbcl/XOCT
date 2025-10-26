"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import torchio as tio


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

def get_params(opt, size, train):
    # (1, 256, 304, 304)
    _, d, h, w = size
    assert opt.preprocess == 'crop'
    
    # we force to use all slices in the depth dimension, because we need to create the projection maps.
    d_ini = 0
    d_end = 0
    
    if train:
        h_ini = random.randint(0, np.maximum(0, h - opt.crop_size))
        w_ini = random.randint(0, np.maximum(0, w - opt.crop_size))
    else:
        h_ini = (h - opt.crop_size) // 2
        w_ini = (w - opt.crop_size) // 2
        
    h_end = h - opt.crop_size - h_ini
    w_end = w - opt.crop_size - w_ini

    flip = random.random() > 0.5

    return {'crop_pos': (d_ini, d_end, h_ini, h_end, w_ini, w_end), 'flip': flip}


def get_transform(opt, params=None):
    transform_list = []
    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(tio.Crop((params['crop_pos'][0], params['crop_pos'][1],params['crop_pos'][2], params['crop_pos'][3],params['crop_pos'][4], params['crop_pos'][5])))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(tio.Flip(axes=2))
    return transforms.Compose(transform_list)



def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
