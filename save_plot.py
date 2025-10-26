import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted

from util.eval import extract_full_proj_from_volume, extract_proj_from_volume


######################################
# 3M
######################################
save_plot_dir = "/home/khan7/workspace/TransPro/eval_plot_transpro_ablation/3M"
os.makedirs(save_plot_dir, exist_ok=True)

res_dir_list = [
    "/home/khan7/workspace/TransPro/octa-500/OCT2OCTA3M_3D/test/A", # gt oct
    "/home/khan7/workspace/TransPro/octa-500/OCT2OCTA3M_3D/test/B", # gt
    "/home/khan7/workspace/TransPro/results/transpro_3M/test_latest",
    "/home/khan7/workspace/TransPro/results/transpro_3M_wo_HCG/test_latest",
    "/home/khan7/workspace/TransPro/results/transpro_3M_wo_VPG/test_latest",
    "/home/khan7/workspace/TransPro/results/transpro_3M_wo_HCG_VPG/test_latest",
    "/home/khan7/workspace/TransPro/results/transpro_3M_wo_HCG_VPG_GAN/test_latest",
    "/home/khan7/workspace/TransPro/results/transpro_3M_3D_Only/test_latest",
    "/home/khan7/workspace/TransPro/results/transpro_3M_3D_Only_wo_GAN/test_latest",
]

seg_dir = "/extra/xielab0/khan7/data/OCTA_500/OCTA_500/Label/GT_Layers_npy"


######################################
# 6M
######################################
# save_plot_dir = "/home/khan7/workspace/TransPro/eval_plot_transpro_ablation/6M"
# os.makedirs(save_plot_dir, exist_ok=True)

# res_dir_list = [
#     "/home/khan7/workspace/TransPro/octa-500/OCT2OCTA6M_3D/test/A", # gt oct
#     "/home/khan7/workspace/TransPro/octa-500/OCT2OCTA6M_3D/test/B", # gt octa
#     "/home/khan7/workspace/TransPro/results/transpro_6M/test_latest",
#     "/home/khan7/workspace/TransPro/results/transpro_6M_wo_HCG/test_latest",
#     "/home/khan7/workspace/TransPro/results/transpro_6M_wo_VPG/test_latest",
#     "/home/khan7/workspace/TransPro/results/transpro_6M_wo_HCG_VPG/test_latest",
#     "/home/khan7/workspace/TransPro/results/transpro_6M_wo_HCG_VPG_GAN/test_latest",
#     "/home/khan7/workspace/TransPro/results/transpro_6M_3D_Only/test_latest",
#     "/home/khan7/workspace/TransPro/results/transpro_6M_3D_Only_wo_GAN/test_latest",
# ]

# seg_dir = "/extra/xielab0/khan7/data/OCTA_500/OCTA_500/Label/GT_Layers_npy"

######################################

title_list = ['gt_oct', 'gt_octa', 'Transpro', 'Transpro_wo_HCG', 'Transpro_wo_VPG', 'Transpro_wo_HCG_VPG', 'Transpro_wo_HCG_VPG_GAN', 'Transpro_3D', 'Transpro_3D_wo_GAN']

fns = natsorted(os.listdir(res_dir_list[0]))
for fn in fns:
    print(f'Processing {fn}')
    vol_list = []
    for i, res_dir in enumerate(res_dir_list):
        res = np.load(os.path.join(res_dir, fn)).astype(np.uint8)
        
        # normalize the ground truth oct and octa
        if i == 0 or i == 1:
            res = ((res-res.min())/(res.max()-res.min()) * 255).astype(np.uint8)
        
        vol_list.append(res)
    
    seg = np.load(os.path.join(seg_dir, fn)).astype(np.uint8)
    
    pid = os.path.splitext(fn)[0]
        
    # mean
    plt.close()
    plt.figure(figsize=(12, 12))
    for i, B in enumerate(vol_list):
        plt.subplot(3, 3, i+1)
        res = B.mean(axis=0)
        if i == 1:
            vmax = res.max() * 1.2
        if i == 0:
            plt.imshow(res, cmap='gray')
        else:
            plt.imshow(res, cmap='gray', vmax=vmax)
        plt.tight_layout()
        plt.title(title_list[i])
    plt.savefig(os.path.join(save_plot_dir, f'{pid}_mean'))
    
    # full_proj
    plt.close()
    plt.figure(figsize=(12, 12))
    for i, B in enumerate(vol_list):
        plt.subplot(3, 3, i+1)
        res = extract_full_proj_from_volume(B, seg, reduction='avg')
        if i == 1:
            vmax = res.max() * 1.2
        if i == 0:
            plt.imshow(res, cmap='gray')
        else:
            plt.imshow(res, cmap='gray', vmax=vmax)
        plt.tight_layout()
        plt.title(title_list[i])
    plt.savefig(os.path.join(save_plot_dir, f'{pid}_full_proj'))
    
    # proj_ILM_OPL
    plt.close()
    plt.figure(figsize=(12, 12))
    for i, B in enumerate(vol_list):
        plt.subplot(3, 3, i+1)
        res = extract_proj_from_volume(B, seg, label=1, reduction='avg')
        if i == 1:
            vmax = res.max() * 1.2
        if i == 0:
            plt.imshow(res, cmap='gray')
        else:
            plt.imshow(res, cmap='gray', vmax=vmax)
        plt.tight_layout()
        plt.title(title_list[i])
    plt.savefig(os.path.join(save_plot_dir, f'{pid}_proj_ILM_OPL'))
    
    # proj_BM_OPL
    plt.close()
    plt.figure(figsize=(12, 12))
    for i, B in enumerate(vol_list):
        plt.subplot(3, 3, i+1)
        res = extract_proj_from_volume(B, seg, label=2, reduction='avg')
        if i == 1:
            vmax = res.max() * 1.2
        if i == 0:
            plt.imshow(res, cmap='gray')
        else:
            plt.imshow(res, cmap='gray', vmax=vmax)
        plt.tight_layout()
        plt.title(title_list[i])
    plt.savefig(os.path.join(save_plot_dir, f'{pid}_proj_OPL_BM'))