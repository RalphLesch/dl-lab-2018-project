#!/bin/bash

cp=$1
shift
mkdir logs/$cp
python3 -u Test_Net.py --model_path=./checkpoints/$cp --dataset_dir=./data/data_CamVidV300/ --logs_path=./logs/$cp/ $@ 2>&1 | tee -a logs/$cp/test.log
