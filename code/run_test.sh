#!/bin/bash

cp=$1
shift
python Test_Net.py --model_path=./checkpoints/$cp --dataset_dir=./data/data_CamVidV300/ $@ 2>&1 | tee logs/${cp}_test.log
