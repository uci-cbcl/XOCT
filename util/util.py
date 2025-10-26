"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)



def central_pad(tensor, target_shape):
    """
    Pads a PyTorch tensor with zeros to achieve the specified target shape, 
    while keeping the original tensor centered.

    Args:
    tensor: Input PyTorch tensor with shape (B, C, D, H, W).
    target_shape: Target shape (d, h, w) as a tuple.

    Returns:
    Padded tensor with the specified target shape.
    """

    b, c, h, w = tensor.shape
    target_h, target_w = target_shape

    # Calculate padding amounts for each dimension
    pad_h = (target_h - h) // 2
    pad_w = (target_w - w) // 2

    # Create padding tuples
    pad_h_tuple = (pad_h, target_h - h - pad_h)
    pad_w_tuple = (pad_w, target_w - w - pad_w)

    # Pad the tensor using torch.nn.functional.pad
    padded_tensor = F.pad(tensor, 
                        pad=[pad_w_tuple[0], pad_w_tuple[1], pad_h_tuple[0], pad_h_tuple[1]], 
                        mode='constant', 
                        value=0) 

    return padded_tensor

def central_crop(tensor, crop_shape):
    """
    Crops the center of a PyTorch tensor to the specified shape.

    Args:
    tensor: Input PyTorch tensor with shape (B, C, D, H, W).
    crop_shape: Target crop shape (d, h, w) as a tuple.

    Returns:
    Cropped tensor with the specified shape.
    """

    b, c, h, w = tensor.shape
    crop_h, crop_w = crop_shape

    # Calculate starting indices for the crop
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2

    # Extract the central crop
    cropped_tensor = tensor[:, :, 
                        start_h:start_h+crop_h, 
                        start_w:start_w+crop_w]

    return cropped_tensor

