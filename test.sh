python test3d_3M.py \
    --dataroot ./octa-500/OCT2OCTA3M_3D \
    --name unet3dmeanstem_3d2d_perceptual_gn_3M \
    --test_name unet3dmeanstem_3d2d_perceptual_gn_3M \
    --model UNet3DMeanStem \
    --direction AtoB \
    --base_ch_g 64 \
    --max_ch_g 512 \
    --dataset_mode alignedoct2octaall \
    --preprocess none \
    --norm group \
    --batch_size 1 \
    --input_nc 1 \
    --output_nc 1 \
    --gpu_ids 0 \
    --num_test 1000000 \
    --which_epoch latest

python eval.py \
    --gt_dir /extra/xielab0/khan7/data/OCTA_500/OCTA_500/OCTA_3mm/OCTA_npy \
    --pred_dir /home/khan7/workspace/OCTA_GAN/results/unet3dmeanstem_3d2d_perceptual_gn_3M/test_latest/fake_B_3d \
    --seg_dir /extra/xielab0/khan7/data/OCTA_500/OCTA_500/Label/GT_Layers_npy \
    --save_dir /home/khan7/workspace/OCTA_GAN/eval_results/unet3dmeanstem_3d2d_perceptual_gn_3M/test_latest/fake_B_3d \
    --csv_fn eval_result.csv \
    --norm \
    --gpu 0

python test3d_6M.py \
    --dataroot ./octa-500/OCT2OCTA6M_3D \
    --name unet3dmeanstem_3d2d_perceptual_gn_6M \
    --test_name unet3dmeanstem_3d2d_perceptual_gn_6M \
    --model UNet3DMeanStem \
    --direction AtoB \
    --base_ch_g 64 \
    --max_ch_g 512 \
    --dataset_mode alignedoct2octaall \
    --preprocess none \
    --norm group \
    --batch_size 1 \
    --input_nc 1 \
    --output_nc 1 \
    --gpu_ids 0 \
    --num_test 1000000 \
    --which_epoch latest

python eval.py \
    --gt_dir /extra/xielab0/khan7/data/OCTA_500/OCTA_500/OCTA_6mm/OCTA_npy \
    --pred_dir /home/khan7/workspace/OCTA_GAN/results/unet3dmeanstem_3d2d_perceptual_gn_6M/test_latest/fake_B_3d \
    --seg_dir /extra/xielab0/khan7/data/OCTA_500/OCTA_500/Label/GT_Layers_npy \
    --save_dir /home/khan7/workspace/OCTA_GAN/eval_results/unet3dmeanstem_3d2d_perceptual_gn_6M/test_latest/fake_B_3d \
    --csv_fn eval_result.csv \
    --norm \
    --gpu 0

# python test3d_3M.py \
#     --dataroot ./octa-500/OCT2OCTA3M_3D \
#     --name unet3d2dsplit3stem_3d2d_perceptual_gn_3M \
#     --test_name unet3d2dsplit3stem_3d2d_perceptual_gn_3M \
#     --model UNet3D2DSplit3Stem \
#     --direction AtoB \
#     --base_ch_g 64 \
#     --max_ch_g 512 \
#     --dataset_mode alignedoct2octaall \
#     --preprocess none \
#     --norm group \
#     --batch_size 1 \
#     --input_nc 1 \
#     --output_nc 1 \
#     --gpu_ids 0 \
#     --num_test 1000000 \
#     --which_epoch latest

# python eval.py \
#     --gt_dir /extra/xielab0/khan7/data/OCTA_500/OCTA_500/OCTA_3mm/OCTA_npy \
#     --pred_dir /home/khan7/workspace/OCTA_GAN/results/unet3d2dsplit3stem_3d2d_perceptual_gn_3M/test_latest/fake_B_3d \
#     --seg_dir /extra/xielab0/khan7/data/OCTA_500/OCTA_500/Label/GT_Layers_npy \
#     --save_dir /home/khan7/workspace/OCTA_GAN/eval_results/unet3d2dsplit3stem_3d2d_perceptual_gn_3M/test_latest/fake_B_3d \
#     --csv_fn eval_result.csv \
#     --norm \
#     --gpu 0

# python test3d_6M.py \
#     --dataroot ./octa-500/OCT2OCTA6M_3D \
#     --name unet3d2dsplit3stem_3d2d_perceptual_gn_6M \
#     --test_name unet3d2dsplit3stem_3d2d_perceptual_gn_6M \
#     --model UNet3D2DSplit3Stem \
#     --direction AtoB \
#     --base_ch_g 64 \
#     --max_ch_g 512 \
#     --dataset_mode alignedoct2octaall \
#     --preprocess none \
#     --norm group \
#     --batch_size 1 \
#     --input_nc 1 \
#     --output_nc 1 \
#     --gpu_ids 0 \
#     --num_test 1000000 \
#     --which_epoch latest

# python eval.py \
#     --gt_dir /extra/xielab0/khan7/data/OCTA_500/OCTA_500/OCTA_6mm/OCTA_npy \
#     --pred_dir /home/khan7/workspace/OCTA_GAN/results/unet3d2dsplit3stem_3d2d_perceptual_gn_6M/test_latest/fake_B_3d \
#     --seg_dir /extra/xielab0/khan7/data/OCTA_500/OCTA_500/Label/GT_Layers_npy \
#     --save_dir /home/khan7/workspace/OCTA_GAN/eval_results/unet3d2dsplit3stem_3d2d_perceptual_gn_6M/test_latest/fake_B_3d \
#     --csv_fn eval_result.csv \
#     --norm \
#     --gpu 0