# FULL
python eval2d.py \
    --gt_dir octa-500/OCT2OCTA6M_3D/Projection_Maps_from_npy/OCTA/norm_True/FULL \
    --pred_dir results/unet2d_gn_6M/test_latest/fake_B_conv_full \
    --save_dir eval_results/unet2d_gn_6M/test_latest/fake_B_conv_full \
    --csv_fn eval_result.csv \
    --norm \
    --gpu 1
