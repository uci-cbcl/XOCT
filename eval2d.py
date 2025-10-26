import numpy as np
import os
import glob
from natsort import natsorted
import argparse
import csv
from tqdm import tqdm
from PIL import Image
import torch
from util.eval import cal_MAE, cal_MSE, cal_PSNR, cal_SSIM
from models.networks_2g_st import VGGPerceptualLoss

def normalize_01(tgt, pred):
    vmin, vmax = tgt.min(), tgt.max()
    tgt = (tgt - vmin) / (vmax - vmin)
    pred = ((pred - vmin) / (vmax - vmin)).clip(0, 1)
    return tgt, pred

def cal_Perceptual(perceptual_model: VGGPerceptualLoss, tgt, pred, norm):
    if norm:
        tgt, pred = normalize_01(tgt, pred)
    with torch.no_grad():
        res = perceptual_model.perceptual_loss(pred, tgt).item()
    return res


def cal_eval_per_patient(perceptual_model, gt_path, pred_path, norm, device):
    gt = np.array(Image.open(gt_path)).astype(np.float32)
    pred = np.array(Image.open(pred_path)).astype(np.float32)
    
    gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)
    pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to(device)
    
    assert gt.shape == pred.shape
        
    eval_res = {}
    
    eval_res['mae'] = cal_MAE(gt, pred, norm=norm)
    eval_res['mse'] = cal_MSE(gt, pred, norm=norm)
    eval_res['psnr'] = cal_PSNR(gt, pred, norm=norm)
    eval_res['ssim'] = cal_SSIM(gt, pred, norm=norm)
    eval_res['perceptual'] = cal_Perceptual(perceptual_model, gt_tensor, pred_tensor, norm=norm)

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
    
    gt_dir = args.gt_dir
    pred_dir = args.pred_dir
    pred_paths = natsorted(glob.glob(os.path.join(pred_dir, '*.bmp')))

    save_dir = args.save_dir
    csv_fn = args.csv_fn
    norm = args.norm
    
    save_dir = os.path.join(save_dir, f'norm_{norm}')
    
    eval_res = {}
    for pred_path in tqdm(pred_paths):
        pid = os.path.basename(pred_path).split('.')[0]
        gt_path = os.path.join(gt_dir, os.path.basename(pred_path))
        
        print(f"Processing {pid}...")
        eval_res[pid] = cal_eval_per_patient(perceptural_model, gt_path, pred_path, norm, device)
    
    attrs = ['pid', 'mae', 'mse', 'psnr', 'ssim', 'perceptual']
    write_csv(eval_res, attrs, save_dir, csv_fn)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--gt_dir', help='gt image directory path', required=True)
    parser.add_argument('--pred_dir', help='pred image directory path', required=True)
    parser.add_argument('--save_dir', help='directory path to save the evaluation result', required=True)
    parser.add_argument('--csv_fn', help='evaluation csv result file name', required=True)
    parser.add_argument('--norm', action='store_true', help="specify this flag if want to do the normalization before the evaluation")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    print(args)
    main(args)
    
