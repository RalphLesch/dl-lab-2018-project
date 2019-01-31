#!/bin/bash

cp=$1
shift
mkdir logs/$cp
stdbuf -o0 -e0 python3 Test_Net.py --model_path=./checkpoints/$cp --dataset_dir=./data/data_CamVidV300/ --logs_path=./logs/$cp/ $@ 2>&1 | tee -a logs/$cp/test.log
