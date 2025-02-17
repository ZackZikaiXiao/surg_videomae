#!/usr/bin/env bash
set -x  # print the commands


MASTER_PORT=12320
GPUS_PER_NODE=8
# Set the path to save checkpoints
OUTPUT_DIR='/home/zikaixiao/zikaixiao/VideoMAE/surg_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800'
# Set the path to Kinetics train set. 
DATA_PATH='/mnt/a100/zikai/surg/surg_videomae/data_util/train_list.csv'

PY_ARGS=${@:1}

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch \
        --nproc_per_node=${GPUS_PER_NODE} \
        --master_port=${MASTER_PORT} \
        run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 32 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 2 \
        --epochs 801 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}