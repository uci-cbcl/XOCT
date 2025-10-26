import time
from options.train_options import TrainOptions
from util.visualizer3d import Logger
from data import create_dataset
from models import create_model
import torch
import numpy as np
from eval3d import cal_SSIM
import random

seed = random.randint(0, 100000000)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def SSIM_(real, fake):
    ssim = cal_SSIM(real[0][0].cpu().numpy(), fake[0][0].cpu().numpy(), norm=True)
    return ssim

def MAE_(fake,real):
    mae = 0.0
    mae = torch.mean(torch.abs(fake-real))
    return mae

def Norm(a):
    max_ = torch.max(a)
    min_ = torch.min(a)
    a_0_1 = (a-min_)/(max_-min_)
    return (a_0_1-0.5)*2

opt = TrainOptions().parse()

dataset = create_dataset(opt, phase="train")  # create a dataset given opt.dataset_mode and other options
dataset_size = len(dataset)
print('#training images = %d' % dataset_size)

val_dataset = create_dataset(opt, phase="val") 
val_dataset_size = len(val_dataset)
print('#validation images = %d' % val_dataset_size)

model = create_model(opt)
model.setup(opt)             
logger = Logger(opt)
total_steps = 0
val_total_iters = 0 

logger.print_line(f"random seed: {seed}")

global_ssim = 0
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    epoch_start_time = time.time()
    iter_start_time = time.time()
    epoch_iter = 0
    if not opt.disable_lr_update:
        model.update_learning_rate()
    for i, data in enumerate(dataset):
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)
        model.optimize_parameters()

        # if total_steps % opt.display_freq == 0:
        #     visualizer.display_current_results(model.get_current_visuals(), epoch)
        #     if "grad" in opt.name:
        #         grads = model.get_current_grads()
        #         visualizer.plot_current_grads(epoch, float(epoch_iter)/dataset_size, opt, grads)
        #         visualizer.print_current_grads(epoch, epoch_iter, grads)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batch_size
            logger.print_current_errors(epoch, epoch_iter, errors, t)
            # if opt.display_id > 0:
            #     visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
    

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
    
    
    if epoch % opt.val_epoch_freq == 0:
        device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        with torch.no_grad():
            SSIM = 0
            SSIM_ILM_OPL = 0
            SSIM_OPL_BM = 0
            SSIM_Mean = 0
            
            num = 0
            for i, data in enumerate(val_dataset):
                model.set_input(data)
                model.test()
                SSIM += model.ssim
                SSIM_ILM_OPL += model.ssim_ilm_opl
                SSIM_OPL_BM += model.ssim_opl_bm
                SSIM_Mean += model.ssim_mean
                num += 1

            logger.print_line(f"Val SSIM: {SSIM/num}; SSIM_ILM_OPL: {SSIM_ILM_OPL/num}; SSIM_OPL_BM: {SSIM_OPL_BM/num}; SSIM_Mean: {SSIM_Mean/num}")
            
            if SSIM/num > global_ssim and epoch > opt.n_epochs:
                global_ssim = SSIM/num
                # Save best models checkpoints
                print('saving the current best model at the end of epoch %d, iters %d' % (epoch, total_steps))
                model.save('best')
                # model.save(epoch)
                print("saving best...")
            # visualizer.print_current_metrics(epoch, MAE/num)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))


#python train3d.py --dataroot /home/slidm/OCTA/octa-500/OCT2OCTA3M_3D --name new_p2p_3D_pm_2g_seg_st_correctfix_seed7 --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --pool_size 0 --load_size 304 --input_nc 1 --output_nc 1 --display_port 6031 --gpu_ids 0 --no_flip --seed 7