import numpy as np
from scipy import special
from matplotlib import cm, pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa

class Augmentation(object):

	def __init__(self, seed=1, aug_type=None, probability=0.9):
		self.type = aug_type
		ia.seed(seed)
		self.shape = [
			iaa.Crop(px=(32, 64)),
			iaa.Fliplr(1),
			iaa.Pad(px=(16, 32), pad_mode=ia.ALL, pad_cval=(0, 128)),
			iaa.CoarseDropout(0.1, size_percent=0.02)
		]
		self.color = [
			iaa.OneOf([
				iaa.GammaContrast((0.75,1.25)),
				iaa.GammaContrast((0.95, 1.05), per_channel=True)
			]),
			# iaa.GaussianBlur(sigma=(1.0, 2.0)),
			iaa.SaltAndPepper(0.005),
			iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.025*255))
		]
		self.seq = self.aug_sequence(aug_type, probability)

	def calculate_probability(self, base_probability, n_elements):
		''' Calculate probablility for each augmentation (or type).'''

		if n_elements == 0:
			return 0
		# calculate coefficients of the polynomial (identical to binomial coefficients) inferred from the inclusion–exclusion principle (P(A or B) = P(B) + P(A) - P(A and B) and P(A) = P(B))
		coefs = [-base_probability] + [special.binom(n_elements, k) * (-1)**(k+1) for k in range(1, n_elements+1)]
		coefs.reverse()

		# calculate real valued roots of the polynomial between 0 and 1 (representing the probability of P(A))

		roots = np.roots(coefs)
		roots = np.real(roots[np.isreal(roots)])
		roots = roots[(roots > 0.0) & (roots < 1.0)]

		return roots[0]

	def aug_sequence(self, type, probability):

		p = self.calculate_probability(0.9, 2)
		p_shape = self.calculate_probability(p, len(self.shape))
		p_color = self.calculate_probability(p, len(self.color))

		seq = []

		if type == "shape":
			seq = [iaa.Sometimes(p_shape, aug) for aug in self.shape]
		if type == "color":
			seq = [iaa.Sometimes(p_color, aug) for aug in self.color]
		if type == "all":
			seq = [iaa.Sometimes(p_shape, aug) for aug in self.shape] + [iaa.Sometimes(p_color, aug) for aug in self.color]

		return iaa.Sequential(seq, random_order=True)

	def np2img(self, np_array):
		return np.clip(np_array, 0, 255)[None,:,:,:]

	def np2segmap(self, np_array, n_classes=12):
		segmap = np_array.astype(np.uint8)[:,:,0]
		return ia.SegmentationMapOnImage(segmap, shape=segmap.shape, nb_classes=n_classes + 1)

	def segmap2np(self, segmap):
		return segmap.get_arr_int().astype(np.float32)[:,:,None]

	def img2np(self, img):
		return img[0]

	def augment_img_and_segmap(self, img, segmap):

		seq_det = self.seq.to_deterministic()
		aug_img = seq_det.augment_images(img)
		aug_segmap = seq_det.augment_segmentation_maps([segmap])[0]

		return aug_img, aug_segmap

	def augment_batch(self, images, segmaps, n_classes=12):

		aug_images = np.zeros(images.shape)
		aug_segmaps = np.zeros(segmaps.shape)

		for (i, (img, segmap)) in enumerate(zip(images, segmaps)):

			img = self.np2img(img)
			segmap = self.np2segmap(segmap, n_classes)

			aug_img, aug_segmap = self.augment_img_and_segmap(img, segmap)

			aug_images[i] = self.img2np(aug_img)
			aug_segmaps[i] = self.segmap2np(aug_segmap)

		return aug_images, aug_segmaps
	#
	# def plot_augmentations(self, img, segmap, n_classes=12, n=4):
	#
	# 	aug_images = []
	# 	aug_segmaps = []
	#
	# 	img = np.clip(img[None,:,:,:] + 128, 0, 255)
	# 	img = img.astype(np.uint8)
	# 	segmap = segmap.astype(np.int32)[:,:,0] + 1
	# 	segmap = ia.SegmentationMapOnImage(segmap, shape=segmap.shape, nb_classes=n_classes)
	#
	# 	for _ in range(n):
	# 		seq_det = self.seq.to_deterministic()
	# 		aug_images.append(seq_det.augment_images(img)[0])
	# 		aug_segmaps.append(seq_det.augment_segmentation_maps([segmap])[0])
	#
	# 	img = img[0]
	#
	# 	cells = []
	# 	for aug_img, aug_segmap in zip(aug_images, aug_segmaps):
	# 		aug_img = aug_img.astype(np.uint8)
	# 		cells.append(img)
	# 		cells.append(segmap.draw(size=aug_img.shape[:2]))
	# 		# cells.append(segmap.draw_on_image(img))
	# 		cells.append(aug_img.astype(np.uint8))
	# 		cells.append(aug_segmap.draw_on_image(aug_img))
	#
	# 	plot = ia.draw_grid(cells, cols=n)
	#
	# 	return plot

	def plot_aug_batch(self, imgs, segmaps, aug_imgs, aug_segmaps, n_classes=12, ncols=12):

		cells = []
		for img, segmap, aug_img, aug_segmap in zip(imgs, segmaps, aug_imgs, aug_segmaps):

			img = self.np2img(img)[0].astype(np.uint8)
			aug_img = self.np2img(aug_img)[0].astype(np.uint8)

			segmap = self.np2segmap(segmap, n_classes)
			aug_segmap = self.np2segmap(aug_segmap, n_classes)

			cells.append(img)
			# cells.append(segmap.draw(size=aug_img.shape[:2]))
			cells.append(segmap.draw_on_image(img))
			cells.append(aug_img)
			cells.append(aug_segmap.draw_on_image(aug_img))

		plot = ia.draw_grid(cells, cols=ncols)

		return plot

def rgb_image(np_array, reverse_colors=False, color_range=(0, 255)):
	'''Converts a numpy array of shape (height, width, 3) into a RGB image.'''

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
