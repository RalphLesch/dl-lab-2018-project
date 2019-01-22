#!/bin/bash

#python Test_Net.py --model_path=./checkpoints/ultraslimS_$1 --configuration=$1 >logs/test$1.log 2>&1 &
python Train_Net.py --checkpoint_dir=./checkpoints/ultraslimS_$1/ --configuration=$1
