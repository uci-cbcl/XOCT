import numpy as np
import os
import glob
from natsort import natsorted
import argparse
import csv
from tqdm import tqdm
from PIL import Image
import torch
from models.networks_2g_st import VGGPerceptualLoss

from util.eval import cal_MAE, cal_MSE, cal_PSNR, cal_SSIM, cal_Perceptual, extract_full_proj_from_volume, extract_proj_from_volume


def cal_eval_per_patient(perceptual_model, gt_path, pred_path, seg_path, norm, device):
    gt_3D = np.load(gt_path)
    pred_3D = np.load(pred_path)[0][0]
    seg_3D = np.load(seg_path)
    
    assert gt_3D.shape == pred_3D.shape and gt_3D.shape == seg_3D.shape, print(gt_3D.shape, pred_3D.shape, seg_3D.shape)
    
    pred_full_proj = extract_full_proj_from_volume(pred_3D, seg_3D, reduction='avg')
    pred_proj_ILM_OPL = extract_proj_from_volume(pred_3D, seg_3D, label=1, reduction='avg')
    pred_proj_OPL_BM = extract_proj_from_volume(pred_3D, seg_3D, label=2, reduction='avg')
    pred_mean_proj = np.mean(pred_3D, axis=0)

    gt_full_proj = extract_full_proj_from_volume(gt_3D, seg_3D, reduction='avg')
    gt_proj_ILM_OPL = extract_proj_from_volume(gt_3D, seg_3D, label=1, reduction='avg')
    gt_proj_OPL_BM = extract_proj_from_volume(gt_3D, seg_3D, label=2, reduction='avg')
    gt_mean_proj = np.mean(gt_3D, axis=0)
    
    eval_res = {}
    
    eval_res['mae_3D'] = cal_MAE(gt_3D, pred_3D, norm=norm)
    eval_res['mse_3D'] = cal_MSE(gt_3D, pred_3D, norm=norm)
    eval_res['psnr_3D'] = cal_PSNR(gt_3D, pred_3D, norm=norm)
    eval_res['ssim_3D'] = cal_SSIM(gt_3D, pred_3D, norm=norm)
    
    eval_res['mae_full_proj'] = cal_MAE(gt_full_proj, pred_full_proj, norm=norm)
    eval_res['mse_full_proj'] = cal_MSE(gt_full_proj, pred_full_proj, norm=norm)
    eval_res['psnr_full_proj'] = cal_PSNR(gt_full_proj, pred_full_proj, norm=norm)
    eval_res['ssim_full_proj'] = cal_SSIM(gt_full_proj, pred_full_proj, norm=norm)
    eval_res['perceptual_full_proj'] = cal_Perceptual(perceptual_model, torch.from_numpy(gt_full_proj).unsqueeze(0).unsqueeze(0).to(device), torch.from_numpy(pred_full_proj).unsqueeze(0).unsqueeze(0).to(device), norm=norm)

    eval_res['mae_proj_ILM_OPL'] = cal_MAE(gt_proj_ILM_OPL, pred_proj_ILM_OPL, norm=norm)
    eval_res['mse_proj_ILM_OPL'] = cal_MSE(gt_proj_ILM_OPL, pred_proj_ILM_OPL, norm=norm)
    eval_res['psnr_proj_ILM_OPL'] = cal_PSNR(gt_proj_ILM_OPL, pred_proj_ILM_OPL, norm=norm)
    eval_res['ssim_proj_ILM_OPL'] = cal_SSIM(gt_proj_ILM_OPL, pred_proj_ILM_OPL, norm=norm)
    eval_res['perceptual_proj_ILM_OPL'] = cal_Perceptual(perceptual_model, torch.from_numpy(gt_proj_ILM_OPL).unsqueeze(0).unsqueeze(0).to(device), torch.from_numpy(pred_proj_ILM_OPL).unsqueeze(0).unsqueeze(0).to(device), norm=norm)

    
    eval_res['mae_proj_OPL_BM'] = cal_MAE(gt_proj_OPL_BM, pred_proj_OPL_BM, norm=norm)
    eval_res['mse_proj_OPL_BM'] = cal_MSE(gt_proj_OPL_BM, pred_proj_OPL_BM, norm=norm)
    eval_res['psnr_proj_OPL_BM'] = cal_PSNR(gt_proj_OPL_BM, pred_proj_OPL_BM, norm=norm)
    eval_res['ssim_proj_OPL_BM'] = cal_SSIM(gt_proj_OPL_BM, pred_proj_OPL_BM, norm=norm)
    eval_res['perceptual_proj_OPL_BM'] = cal_Perceptual(perceptual_model, torch.from_numpy(gt_proj_OPL_BM).unsqueeze(0).unsqueeze(0).to(device), torch.from_numpy(pred_proj_OPL_BM).unsqueeze(0).unsqueeze(0).to(device), norm=norm)
    
    eval_res['mae_mean_proj'] = cal_MAE(gt_mean_proj, pred_mean_proj, norm=norm)
    eval_res['mse_mean_proj'] = cal_MSE(gt_mean_proj, pred_mean_proj, norm=norm)
    eval_res['psnr_mean_proj'] = cal_PSNR(gt_mean_proj, pred_mean_proj, norm=norm)
    eval_res['ssim_mean_proj'] = cal_SSIM(gt_mean_proj, pred_mean_proj, norm=norm)
    eval_res['perceptual_mean_proj'] = cal_Perceptual(perceptual_model, torch.from_numpy(gt_mean_proj).unsqueeze(0).unsqueeze(0).to(device), torch.from_numpy(pred_mean_proj).unsqueeze(0).unsqueeze(0).to(device), norm=norm)
    
    
    
    return eval_res

def write_csv(eval_res, attrs, save_dir, csv_fn):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, csv_fn), 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(attrs)
        for pid, eval_per_pid in eval_res.items():
            csv_row = []
            csv_row.append(pid)
            for attr in attrs[1:]:
                csv_row.append(eval_per_pid[attr])
            csv_writer.writerow(csv_row)
        

def main(args):
    device = torch.device(f"cuda:{args.gpu}")
    perceptural_model = VGGPerceptualLoss(layers=["relu3_3"], requires_grad=False).to(device)
    # Create a pool of worker processes
    
    gt_dir = args.gt_dir

    pred_dir = args.pred_dir
    pred_paths = natsorted(glob.glob(os.path.join(pred_dir, '*.npy')))

    seg_dir = args.seg_dir
    save_dir = args.save_dir
    csv_fn = args.csv_fn
    norm = args.norm
    
    save_dir = os.path.join(save_dir, f'norm_{norm}')
    
    eval_res = {}
    for pred_path in tqdm(pred_paths):
        pid = os.path.basename(pred_path).split('.')[0]
        gt_path = os.path.join(gt_dir, os.path.basename(pred_path))
        
        print(f"Processing {pid}...")
        seg_path = os.path.join(seg_dir, os.path.basename(gt_path))
        async_result = cal_eval_per_patient(perceptural_model, gt_path, pred_path, seg_path, norm, device)
        eval_res[pid] = async_result
    
    
        
    attrs = ['pid', 
             'mae_3D', 'mse_3D', 'psnr_3D', 'ssim_3D', \
             'mae_full_proj', 'mse_full_proj', 'psnr_full_proj', 'ssim_full_proj', 'perceptual_full_proj', \
             'mae_proj_ILM_OPL', 'mse_proj_ILM_OPL', 'psnr_proj_ILM_OPL', 'ssim_proj_ILM_OPL', 'perceptual_proj_ILM_OPL', \
             'mae_proj_OPL_BM', 'mse_proj_OPL_BM', 'psnr_proj_OPL_BM', 'ssim_proj_OPL_BM', 'perceptual_proj_OPL_BM', \
            'mae_mean_proj', 'mse_mean_proj', 'psnr_mean_proj', 'ssim_mean_proj', 'perceptual_mean_proj']
    
    write_csv(eval_res, attrs, save_dir, csv_fn)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--gt_dir', help='gt image directory path', required=True)
    parser.add_argument('--pred_dir', help='pred image directory path', required=True)
    parser.add_argument('--seg_dir', help='layer segmentation image directory path', required=True)
    parser.add_argument('--save_dir', help='directory path to save the evaluation result', required=True)
    parser.add_argument('--csv_fn', help='evaluation csv result file name', required=True)
    parser.add_argument('--norm', action='store_true', help="specify this flag if want to do the normalization before the evaluation")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    print(args)
    main(args)
    
