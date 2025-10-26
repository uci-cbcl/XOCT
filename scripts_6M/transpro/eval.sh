python eval3d.py \
    --gt_dir /extra/xielab0/khan7/data/OCTA_500/OCTA_500/OCTA_6mm/OCTA_npy \
    --pred_dir results/transpro_gn_6M/test_latest/fake_B_3d \
    --seg_dir ./octa-500/Label/GT_Layers_npy \
    --save_dir eval_results/transpro_gn_6M/test_latest/fake_B_3d \
    --csv_fn eval_result.csv \
    --norm \
    --gpu 0