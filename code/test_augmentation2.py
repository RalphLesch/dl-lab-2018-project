import numpy as np
import tensorflow as tf
import Augmentation
from matplotlib import cm, pyplot as plt


mmap_mode='r'

path = './data/'

data = np.load(path + 'Train_data_CamVid.npy', mmap_mode=mmap_mode)
label = np.load(path + 'Train_label_CamVid.npy', mmap_mode=mmap_mode)

class_range = (np.min(label), np.max(label))


#########################################

with tf.Session() as sess:
	aug = Augmentation.Augmentation(aug_type='all', probability=0.999, seed=1)
	aug.tf_session = sess

	n_rows = 5
	n_cols = 4
	#fig, axs = plt.subplots(nrows=n, ncols=4)
	fig = plt.figure(figsize=(8,10))
	for i in range(0, n_rows):

		d, l, info = aug.augment_img(data[i], label[i]) # TODO: copy should not be required!

		print('image ' + str(i) + ' - augmentation types: ' + ', '.join(info.keys()))

		pos = n_cols*i+1
		fig.add_subplot(n_rows, n_cols, pos, title='image ' + str(i)).axis('off')
		#plt.imshow(data[i].transpose((1, 2, 0)), cmap='gray')
		aug.rgb_image(data[i])
		pos = n_cols*i+2
		fig.add_subplot(n_rows, n_cols, pos, title='label').axis('off')
		aug.class_image(label[i])
		#aug.class_image(None, label[i])

		pos = n_cols*i+3
		fig.add_subplot(n_rows, n_cols, pos, title='aug img').axis('off')
		aug.rgb_image(d)
		pos = n_cols*i+4
		fig.add_subplot(n_rows, n_cols, pos, title='aug label').axis('off')
		aug.class_image(l)

		#aug.class_image(None, l)
	fig.tight_layout()
	plt.subplots_adjust(hspace=0.2, wspace=0)
	plt.show()
