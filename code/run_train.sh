#!/bin/bash

# screen -S t1
# source ../../venv3/bin/activate
# python3 run_train.sh <subfolder: checkpoint and log> --aug_type=<type>

#python Train_Net.py --checkpoint_dir=./checkpoints/ultraslimS_$1/ --configuration=$1
#python Train_Net.py --dataset_dir=./data/data_CamVidV300/ --checkpoint_dir=./checkpoints/$2/ --max_steps=$3 --augmentation=$4
cp=$1
shift

mkdir logs/$cp
stdbuf -o0 -e0 python3 Train_Net.py --dataset_dir=./data/data_CamVidV300/ --checkpoint_dir=./checkpoints/$cp/ --logs_path=./logs/$cp/ --max_steps=21000 $@ 2>&1 | tee logs/$cp/train.log
