from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import torch.nn.functional as F


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().detach().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def tensor2im3d(image_tensor):
    image_numpy = image_tensor[0][0].cpu().float().numpy()
    return image_numpy

def mask2im(image_tensor, imtype=np.uint8):
    image_numpy = F.one_hot(image_tensor.cpu().float().detach().argmax(dim=0), 2).permute(2, 0, 1).numpy()
    image_numpy = np.expand_dims(np.argmax(image_numpy, axis=0)*255,axis=2)
    return image_numpy.astype(imtype)

def diagnose_network(net, name='network'):
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


def save_image3d(image_numpy, image_path):
    for i in range (image_numpy.shape[1]):
        img_arr = image_numpy[i,:,:]
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = np.tile(img_arr, (3, 1, 1))
        
        img_arr = (np.transpose(img_arr, (1, 2, 0)) + 1) / 2.0 * 255.0
        img_arr = img_arr.astype(np.uint8)
        image_pil = Image.fromarray(img_arr)
        fn = image_path+str(i)+".png"
        image_pil.save(fn)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
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

    b, c, d, h, w = tensor.shape
    target_d, target_h, target_w = target_shape

    # Calculate padding amounts for each dimension
    pad_d = (target_d - d) // 2 
    pad_h = (target_h - h) // 2
    pad_w = (target_w - w) // 2

    # Create padding tuples
    pad_d_tuple = (pad_d, target_d - d - pad_d) 
    pad_h_tuple = (pad_h, target_h - h - pad_h)
    pad_w_tuple = (pad_w, target_w - w - pad_w)

    # Pad the tensor using torch.nn.functional.pad
    padded_tensor = F.pad(tensor, 
                        pad=[pad_w_tuple[0], pad_w_tuple[1], pad_h_tuple[0], pad_h_tuple[1], pad_d_tuple[0], pad_d_tuple[1]], 
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

    b, c, d, h, w = tensor.shape
    crop_d, crop_h, crop_w = crop_shape

    # Calculate starting indices for the crop
    start_d = (d - crop_d) // 2
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2

    # Extract the central crop
    cropped_tensor = tensor[:, :, start_d:start_d+crop_d, 
                        start_h:start_h+crop_h, 
                        start_w:start_w+crop_w]

    return cropped_tensor
