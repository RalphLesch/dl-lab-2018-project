#!/bin/bash

python3 plot.py --tests --noplot --output plots/ $@
python3 plot.py --tests --noplot --merge --output plots/compare $@
python3 plot.py --tests ../code/test/imgaug*/  --noplot --merge --output plots/comp_imgaug $@

