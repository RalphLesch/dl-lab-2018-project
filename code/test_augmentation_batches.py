import numpy as np
import Augmentation
from matplotlib import cm, pyplot as plt

import json
import sys

mmap_mode='r'

path = 'data/data_CamVidV300/'

data = np.load(path + 'Train_data_CamVid.npy', mmap_mode=mmap_mode)
label = np.load(path + 'Train_label_CamVid.npy', mmap_mode=mmap_mode)

#print(np.min(data), np.max(data))

#data = data + 128


#########################################

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
aug = Augmentation.Augmentation(aug_type='all', probability=0.9, seed=seed)

n_rows = 5
n_cols = 4

d_batch = data[0:n_rows*10] + 128
l_batch = label[0:n_rows*10]

# aug.augment_batch(data[1:10,:,:,:], label[1:10,:,:])
d_aug_batch, l_aug_batch, info = aug.augment_batch(d_batch, l_batch)

#fig, axs = plt.subplots(nrows=n, ncols=4)
fig = plt.figure(figsize=(8,10))
stats = []
for i in range(0, n_rows):
	
	j = i*10
	d = d_batch[j]
	l = l_batch[j]
	d_aug = d_aug_batch[j]
	l_aug = l_aug_batch[j]
	current_info = info[j]
	
	stats.append(current_info)

	print('image ' + str(j) + ' - augmentation types: ' + ', '.join(current_info.keys()))
	print(current_info)

	pos = n_cols*i+1
	fig.add_subplot(n_rows, n_cols, pos, title='image ' + str(j)).axis('off')
	#plt.imshow(d.transpose((1, 2, 0)), cmap='gray')
	aug.rgb_image(d, color_range=(0, 255))
	pos = n_cols*i+2
	fig.add_subplot(n_rows, n_cols, pos, title='label').axis('off')
	aug.class_image(l)
	#aug.class_image(None, l)

	pos = n_cols*i+3
	fig.add_subplot(n_rows, n_cols, pos, title='aug img').axis('off')
	aug.rgb_image(d_aug, color_range=(0, 255))
	pos = n_cols*i+4
	fig.add_subplot(n_rows, n_cols, pos, title='aug label').axis('off')
	aug.class_image(l_aug)

	#aug.class_image(None, l)
fig.tight_layout()
plt.subplots_adjust(hspace=0.2, wspace=0)
#print(stats)
with open('stats.json', 'w') as file:
	json.dump(stats, file)
plt.show()
