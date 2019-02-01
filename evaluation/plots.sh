#!/bin/bash

python3 plot.py --tests --noplot --output plots/ $@
python3 plot.py --tests --noplot --merge --output plots/compare $@
python3 plot.py --tests ../code/tests/imgaug*/ ../code/tests/aug_hack*/  --noplot --merge --output plots/comp_aug_hack-imgaug $@

