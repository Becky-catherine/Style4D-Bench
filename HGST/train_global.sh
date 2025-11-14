#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu_4090
#SBATCH --output=./log/output_global.log 
python train_new.py --style_dir ./train_1 --content_dir ./train2014 --save_dir models_global/ --batch_size 4