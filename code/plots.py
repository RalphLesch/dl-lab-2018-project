import numpy as np
import imgaug as ia
from matplotlib import cm, pyplot as plt
from imgaug_augmentation import Augmentation as aug

def plot_aug_batch(imgs, segmaps, aug_imgs, aug_segmaps, n_classes=12, ncols=12):

	cells = []
	for img, segmap, aug_img, aug_segmap in zip(imgs, segmaps, aug_imgs, aug_segmaps):

		img = aug.np2img(img)[0].astype(np.uint8)
		aug_img = aug.np2img(aug_img)[0].astype(np.uint8)

		segmap = aug.np2segmap(segmap, n_classes)
		aug_segmap = aug.np2segmap(aug_segmap, n_classes)

		cells.append(img)
		# cells.append(segmap.draw(size=aug_img.shape[:2]))
		cells.append(segmap.draw_on_image(img, alpha=0.5))
		cells.append(aug_img)
		cells.append(aug_segmap.draw_on_image(aug_img, alpha=0.5))

	plot = ia.draw_grid(cells, cols=ncols)

	return plot

def plot_prediction(imgs, segmaps, predictions, n_classes=12, ncols=3):

	cells = []
	for img, segmap, pred in zip(imgs, segmaps, predictions):

		img = aug.np2img(img)[0].astype(np.uint8)
		segmap = aug.np2segmap(segmap, n_classes)
		pred = aug.np2segmap(pred, n_classes)

		cells.append(img)
		# cells.append(segmap.draw(size=aug_img.shape[:2]))
		cells.append(segmap.draw_on_image(img, alpha=0.5))
		cells.append(pred.draw_on_image(img, alpha=0.5))

	plot = ia.draw_grid(cells, cols=ncols)

	return plot

def rgb_image(np_array, reverse_colors=False, color_range=None):
	'''Converts a numpy array of shape (height, width, 3) into a RGB image.'''

	if color_range is None:
		color_range = (np.min(np_array), np.max(np_array))

	np_array = np_array.copy()

	n_colors = color_range[1] - color_range[0] + 1

	# normalize colors into range 0 to 1
	np_array -= color_range[0]
	np_array /= n_colors

	# flip color channels
	if reverse_colors:
		np_array = np_array[:,:,::-1]

	return plt.imshow(np_array)

def class_image(np_array, class_range=(0,12)):
	'''Converts a numpy array of shape (height, width, 1) into an image with different colors for different classes.'''

	# calculate range of different class labels
	if class_range is None:
		class_range = (np.min(np_array), np.max(np_array))

	n_classes = class_range[1] - class_range[0] + 1

	np_array = np_array.copy()

	# normalize classes into range 0 to 1
	np_array -= class_range[0]
	np_array /= n_classes

	# define colormap to distinguish different classes
	colormap = cm.get_cmap(name='nipy_spectral', lut=n_classes).reversed()

	return plt.imshow(np_array[:,:,0], cmap=colormap)
