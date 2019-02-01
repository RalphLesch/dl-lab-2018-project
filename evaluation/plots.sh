#!/bin/bash

python3 plot.py --checkpoints --noplot --output plots/ $@
python3 plot.py --checkpoints --noplot --merge --output plots/compare $@

