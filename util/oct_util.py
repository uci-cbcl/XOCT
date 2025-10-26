import numpy as np

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