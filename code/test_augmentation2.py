import numpy as np
import tensorflow as tf
import Augmentation
from matplotlib import cm, pyplot as plt
from skimage import exposure, color

aug = Augmentation.Augmentation(aug_type='all', probability=0.999, seed=1)
mmap_mode='r'

path = 'data/data_CamVidV300/'

data = np.load(path + 'Train_data_CamVid.npy', mmap_mode=mmap_mode)
label = np.load(path + 'Train_label_CamVid.npy', mmap_mode=mmap_mode)

# image = data[0,:,:,:] + 128
# aug.rgb_image(image - 128)
# plt.show()
#
#
# hsv = color.rgb2hsv(image)
# hsv[:, :, 0] = -20
# image = color.hsv2rgb(hsv)
# aug.rgb_image(image - 128)
# plt.show()

print(np.min(data), np.max(data))

data = data + 128

# exposure.adjust_gamma(data[0,:,:,:], 2)
#
# print(np.min(data[0,:,:,:]))

class_range = (np.min(label), np.max(label))


#########################################

with tf.Session() as sess:
	aug = Augmentation.Augmentation(aug_type='all', probability=0.9, seed=1)
	aug.tf_session = sess

	n_rows = 5
	n_cols = 4
	#fig, axs = plt.subplots(nrows=n, ncols=4)
	fig = plt.figure(figsize=(8,10))
	stats = []
	for i in range(0, n_rows):

		# aug.augment_batch(data[1:10,:,:,:], label[1:10,:,:])
		d, l, info = aug.augment_img(data[i], label[i]) # TODO: copy should not be required!
		stats.append(info)

		print('image ' + str(i) + ' - augmentation types: ' + ', '.join(info.keys()))

		pos = n_cols*i+1
		fig.add_subplot(n_rows, n_cols, pos, title='image ' + str(i)).axis('off')
		#plt.imshow(data[i].transpose((1, 2, 0)), cmap='gray')
		aug.rgb_image(data[i], color_range=(0, 255))
		pos = n_cols*i+2
		fig.add_subplot(n_rows, n_cols, pos, title='label').axis('off')
		aug.class_image(label[i])
		#aug.class_image(None, label[i])

		pos = n_cols*i+3
		fig.add_subplot(n_rows, n_cols, pos, title='aug img').axis('off')
		aug.rgb_image(d, color_range=(0, 255))
		pos = n_cols*i+4
		fig.add_subplot(n_rows, n_cols, pos, title='aug label').axis('off')
		aug.class_image(l)

		#aug.class_image(None, l)
	fig.tight_layout()
	plt.subplots_adjust(hspace=0.2, wspace=0)
	print(stats)
	plt.show()
