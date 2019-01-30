#!/bin/bash

cp=$1
shift
python Test_Net.py --model_path=./checkpoints/$cp --dataset_dir=./data/data_CamVidV300/ --logs_path=./logs/$cp/ $@ 2>&1 | tee logs/$cp/test.log
