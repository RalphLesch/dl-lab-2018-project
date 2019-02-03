#!/bin/bash

plot_cmd() {
python3 plot.py --tests ../code/test/{imgaug_none,imgaug_shape,imgaug_color,imgaug} ../code/test/{imgaug_none,imgaug_shape,imgaug_color,imgaug}_small \
 --labels 'no augment (I)' 'shape (I)' 'color (I)' 'shape & color (I)' 'no augment (II) ' 'shape (II)' 'color (II)' 'shape & color (II)' \
 --linestyle '- - - - -- -- -- --' \
 -x steps -y IoU \
 --reset-color-index 4 \
 --merge --noplot -f pdf $@
}

plot_cmd --output plots/result
plot_cmd --output plots/result_legend_below --legendbelow 2

