#!/bin/bash

python3 plot.py --tests ../code/test/{imgaug_none,imgaug_shape,imgaug_color,imgaug} ../code/test/{imgaug_none,imgaug_shape,imgaug_color,imgaug}_small \
 --labels 'no augment (I)' 'shape (I)' 'color (I)' 'shape & color (I)' 'no augment (II) ' 'shape (II)' 'color (II)' 'shape & color (II)' \
 --linestyle '- - - - -- -- -- --' \
 -x steps -y IoU \
 --merge --output plots/result \
 --legendbelow 2 \
 --noplot -f pdf

