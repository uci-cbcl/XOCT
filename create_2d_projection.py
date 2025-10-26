import os
from natsort import natsorted
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

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
    
    
def normalize(img):
    vmin, vmax = img.min(), img.max()
    img = (img - vmin) / (vmax - vmin) * 255
    return img

def main(args):
    vol_dir = args.vol_dir
    seg_dir = args.seg_dir
    norm = args.norm
    save_dir = args.save_dir
    
    save_dir = os.path.join(save_dir, f'norm_{norm}')
    full_proj_save_fir = os.path.join(save_dir, 'FULL')
    mean_proj_save_fir = os.path.join(save_dir, 'Mean')
    proj_ILM_OPL_save_fir = os.path.join(save_dir, 'ILM_OPL')
    proj_OPL_BM_save_fir = os.path.join(save_dir, 'OPL_BM')

    os.makedirs(full_proj_save_fir, exist_ok=True)
    os.makedirs(mean_proj_save_fir, exist_ok=True)
    os.makedirs(proj_ILM_OPL_save_fir, exist_ok=True)
    os.makedirs(proj_OPL_BM_save_fir, exist_ok=True)

    for vol_fn in natsorted(os.listdir(vol_dir)):
        seg_path = os.path.join(seg_dir, vol_fn)
        assert os.path.exists(seg_path)
        
    for vol_fn in tqdm(natsorted(os.listdir(vol_dir))):
        vol_path = os.path.join(vol_dir, vol_fn)
        seg_path = os.path.join(seg_dir, vol_fn)
        
        vol_3D = np.load(vol_path)
        seg_3D = np.load(seg_path)
        
        full_proj = extract_full_proj_from_volume(vol_3D, seg_3D, reduction='avg')
        proj_ILM_OPL = extract_proj_from_volume(vol_3D, seg_3D, label=1, reduction='avg')
        proj_OPL_BM = extract_proj_from_volume(vol_3D, seg_3D, label=2, reduction='avg')
        mean_proj = np.mean(vol_3D, axis=0)
        
        if norm:
            full_proj = normalize(full_proj)
            mean_proj = normalize(mean_proj)
            proj_ILM_OPL = normalize(proj_ILM_OPL)
            proj_OPL_BM = normalize(proj_OPL_BM)
        
        full_proj = full_proj.astype(np.uint8)
        mean_proj = mean_proj.astype(np.uint8)
        proj_ILM_OPL = proj_ILM_OPL.astype(np.uint8)
        proj_OPL_BM = proj_OPL_BM.astype(np.uint8)
            
        full_proj = Image.fromarray(full_proj)
        mean_proj = Image.fromarray(mean_proj)
        proj_ILM_OPL = Image.fromarray(proj_ILM_OPL)
        proj_OPL_BM = Image.fromarray(proj_OPL_BM)
        
        full_proj.save(os.path.join(full_proj_save_fir, os.path.splitext(vol_fn)[0] + '.bmp'))
        mean_proj.save(os.path.join(mean_proj_save_fir, os.path.splitext(vol_fn)[0] + '.bmp'))
        proj_ILM_OPL.save(os.path.join(proj_ILM_OPL_save_fir, os.path.splitext(vol_fn)[0] + '.bmp'))
        proj_OPL_BM.save(os.path.join(proj_OPL_BM_save_fir, os.path.splitext(vol_fn)[0] + '.bmp'))
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--vol_dir', help='gt image directory path', required=True)
    parser.add_argument('--seg_dir', help='layer segmentation image directory path', required=True)
    parser.add_argument('--save_dir', help='directory path to save the evaluation result', required=True)
    parser.add_argument('--norm', action='store_true', help="specify this flag if want to do the normalization before the evaluation")
    args = parser.parse_args()
    
    print(args)
    main(args)
    