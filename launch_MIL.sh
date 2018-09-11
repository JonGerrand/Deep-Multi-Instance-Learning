#! /bin/bash

## -----------
# This script is intended to perform the complete MIL training process for a
# given dataset provided.
#
# The data should be stored in '/data/dataset/raw' with the 'positive' class
# data being placed in a folder named 'd_1' and the 'negative' class in the
# 'd_0' folder.
#
# Author: Jonathan Gerrand
## -----------

# --- Setup configs ---
model=GoogleNet #GoogleNet/NIN
model_data_path=data/models/googlenet/GoogleNet.npy # path/to/your/model/files

dataset_dir=data/dataset # path/to/dataset/dir
checkpoint_dir=data/checkpoint # path/to/your/checkpoint/dump/dir

# MIL Image Sampling
crop_step_size=224
raw_img_len=448
resize_img_len=224
patch_type=grid_cent

# Training Params
epochs=60
batch_size=4
optimiser=Adam
reinit_learn_rate=0.01
fine_tune_learn_rate=0.001

# ---Stage 1 Folder Prep---
echo '** Setting up for Stage 1 **'
if [ ! -d $checkpoint_dir ]; then
  mkdir $checkpoint_dir
fi
Stage_1_model_ckp=$checkpoint_dir/stage_1
if [ ! -d $Stage_1_model_ckp ]; then
  mkdir $Stage_1_model_ckp
fi
Stage_1_data_dir=$dataset_dir

python prep/prepare_dataset.py \
  --root_dir $dataset_dir

# ---Stage 1 Training---
echo '** Performing Stage 1 Training **'
python mil/mil_training.py \
  --model_name $model \
  --train_data_path $Stage_1_data_dir \
  --model_data_path $model_data_path \
  --checkpoint_path $Stage_1_model_ckp \
  --train_type stage_1 \
  --init_type Fine-tune \
  --optimiser $optimiser \
  --reinit_learn_rate $reinit_learn_rate \
  --fine_tune_learn_rate $fine_tune_learn_rate \
  --batch_size $batch_size \
  --num_epochs $epochs \
  --weight_decay 0.0 \
  --val_metric score \
  --raw_img_len $raw_img_len \
  --model_img_len $resize_img_len \
  --crop_step_size $crop_step_size \
  --patch_type $patch_type
echo
echo '** Stage 1 Training Completed **'

# ---Stage 2 Folder Prep---
echo '** Setting up for stage 2 **'
Stage_2_model_ckp=$checkpoint_dir/stage_2/
if [ ! -d $Stage_2_model_ckp ]; then
  mkdir $Stage_2_model_ckp
fi
Stage_2_data_dir=$dataset_dir/stage_2/
if [ ! -d $Stage_2_data_dir ]; then
  mkdir $Stage_2_data_dir
  mkdir $Stage_2_data_dir/raw
fi

echo '** Extracting discriminative image patches **'
python mil/extract_discriminative.py \
  --model_name $model \
  --img_data_path $Stage_1_data_dir \
  --model_data_path $Stage_1_model_ckp \
  --save_root_path $Stage_2_data_dir/raw/ \
  --subset_type train \
  --raw_img_len $raw_img_len \
  --model_img_len $resize_img_len \
  --crop_step_size $crop_step_size \
  --patch_type $patch_type  \
  --thresh_val 0.6

python prep/prepare_dataset.py \
  --root_dir $Stage_2_data_dir \
  --stage 'stage_2'

# ---Stage 2 Training---
echo '** Performing Stage 2 Training **'
python mil/mil_training.py \
  --model_name $model \
  --train_data_path $Stage_2_data_dir \
  --model_data_path $Stage_1_model_ckp \
  --checkpoint_path $Stage_2_model_ckp \
  --train_type stage_2 \
  --init_type Multi-fine-tune \
  --optimiser $optimiser \
  --reinit_learn_rate $reinit_learn_rate \
  --fine_tune_learn_rate $(echo "scale=7; $fine_tune_learn_rate/10" | bc) \
  --batch_size $batch_size \
  --raw_img_len $raw_img_len \
  --crop_step_size $crop_step_size \
  --val_metric score \
  --patch_type $patch_type

echo '** Stage 2 Training Completed **'

# ---Stage 2 Testing---
echo '** Testing Model **'
python mil/mil_testing.py \
  --model_name $model \
  --train_data_path $Stage_1_data_dir \
  --model_data_path $Stage_2_model_ckp \
  --subset_type 'test' \
  --testing_type indicator \
  --raw_img_len $raw_img_len \
  --model_img_len $resize_img_len \
  --crop_step_size $crop_step_size \
  --thresh 0.5 \
  --roc_curve \
  --patch_type $patch_type > $Stage_2_model_ckp/results.txt
