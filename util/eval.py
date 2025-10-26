import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import torch
from models.networks_2g_st import VGGPerceptualLoss

def extract_full_proj_from_volume(volume, seg, reduction):
    v = volume * (seg > 0)
    if reduction == 'max':
        return np.max(v, axis=0, keepdims=False)
    elif reduction == 'avg':
        return np.sum(v, axis=0, keepdims=False) / (np.sum(seg > 0, axis=0, keepdims=False) + 1e-6)
    else:
        raise NotImplementedError
    
def extract_proj_from_volume(volume, seg, label, reduction):
    v = volume * (seg == label)
    if reduction == 'max':
        return np.max(v, axis=0, keepdims=False)
    elif reduction == 'avg':
        return np.sum(v, axis=0, keepdims=False) / (np.sum(seg == label, axis=0, keepdims=False) + 1e-6)
    else:
        raise NotImplementedError
    
def normalize(tgt, pred):
    vmin, vmax = tgt.min(), tgt.max()
    tgt = (tgt - vmin) / (vmax - vmin) * 255
    pred = ((pred - vmin) / (vmax - vmin)).clip(0, 1) * 255
    return tgt, pred

def cal_MAE(tgt, pred, norm=False):
    if norm:
        tgt, pred = normalize(tgt, pred)
    mae = np.mean(np.abs(tgt - pred))
    return mae

def cal_MSE(tgt, pred, norm=False):
    if norm:
        tgt, pred = normalize(tgt, pred)
    mse = np.mean((tgt - pred) ** 2)
    return mse

def cal_SSIM(tgt, pred, norm=False):
    if norm:
        tgt, pred = normalize(tgt, pred)
    ssim = structural_similarity(tgt, pred, data_range=tgt.max() - tgt.min())
    return ssim

def cal_PSNR(tgt, pred, norm=False):
    if norm:
        tgt, pred = normalize(tgt, pred)
    psnr = peak_signal_noise_ratio(tgt, pred, data_range=tgt.max() - tgt.min())
    return psnr

def cal_Perceptual(perceptual_model: VGGPerceptualLoss, tgt, pred, norm):
    if norm:
        tgt, pred = normalize(tgt, pred)
        tgt = tgt / 255
        pred = pred / 255
    with torch.no_grad():
        res = perceptual_model.perceptual_loss(pred.float(), tgt.float()).item()
    return res
