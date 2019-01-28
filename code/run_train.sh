#!/bin/bash

#python Train_Net.py --checkpoint_dir=./checkpoints/ultraslimS_$1/ --configuration=$1
python Train_Net.py --checkpoint_dir=./checkpoints/$1/ --dataset_dir=./data/$1/ --max_steps=2000
