import random
import numpy as np
from scipy import special
from matplotlib import cm, pyplot as plt
import cv2 as cv
from skimage import exposure, color, img_as_float
from RandomParam import RandomParam

class Augmentation(object):

	def __init__(self, aug_type=None, probability=0.9, seed=None):
		self.augmentations = {'shape': Shape(self), 'color': Color(self)}
		self.type = aug_type
		self.probability = probability
		# self.random = random.Random()
		# if seed is not None:
		# 	self.random.seed(seed)
		self.rand = RandomParam(seed)


	@property
	def type(self):
		return self._type

	@type.setter
	def type(self, aug_type):
		if not (aug_type in [None, 'all', *self.augmentations]):
			raise ValueError("type '{}' is invalid, must be one of {}".format(aug_type, [None, 'all', *self.augmentations]))
		self._type = aug_type

	def augment_batch(self, data, label):
		'''Apply random augmentations to all the images of the data and label batches.'''

		N, height, width, channels = data.shape

		augmentation_infos = []

		aug_data = np.zeros(data.shape, data.dtype)
		aug_label = np.zeros(label.shape, label.dtype)

		for i in range(N):
			# print('image', i)
			aug_data[i,:,:,:], aug_label[i,:,:,:], infos = self.augment_img(data[i,:,:,:], label[i,:,:,:])
			augmentation_infos.append(infos)

		return data, label, augmentation_infos


	def augment_img(self, data, label):
		'''Augment the data and label with the set probability and type.'''

		#print('AUGMENTATION: ', self.type)

		infos = dict()

		if self._type is None:
			return data, label, infos
		if self._type == 'all':
			# TODO: include no augmentation
			# Probability per typ
			p_type = self.calculate_probability(self.probability, len(self.augmentations))
			for t in self.augmentations:
				data, label, info = self.random_augmentation_of_type(t, data, label, p_type)
				infos.update(info)
		else:
			data, label, infos = self.random_augmentation_of_type(self._type, data, label, self.probability)

		return data, label, infos


	def calculate_probability(self, base_probability, n_elements):
		''' Calculate probablility for each augmentation (or type).'''

		# calculate coefficients of the polynomial (identical to binomial coefficients) inferred from the inclusion–exclusion principle (P(A or B) = P(B) + P(A) - P(A and B) and P(A) = P(B))
		coefs = [-base_probability] + [special.binom(n_elements, k) * (-1)**(k+1) for k in range(1, n_elements+1)]
		coefs.reverse()

		# calculate real valued roots of the polynomial between 0 and 1 (representing the probability of P(A))
		roots = np.roots(coefs)
		roots = np.real(roots[np.isreal(roots)])
		roots = roots[(roots > 0.0) & (roots < 1.0)]

		return roots[0]

	def random_augmentation_of_type(self, aug_type, data, label, probability):
		''' Apply random augmentation of the given type (see augmentations) to data and label, with the given base probability.'''

		t = self.augmentations[aug_type]
		# Probability for each augmentation.
		p_augment = self.calculate_probability(probability, len(t))
		augmentations = list(t.keys())
		self.rand.shuffle(augmentations)

		# save informations about the augmentation types and their parameters
		info = dict()

		for a_index in augmentations:
			augmentation = t[a_index]
			i = self.rand.random_unif(1, 0, 1)[0]
			if i < p_augment:
				# print('    augment:', a_index)
				data, label, params = augmentation(data, label)
				info[a_index] = params

		return data, label, info

	# ------------------------------------------------------------------------------
	# helper functions to plot a rgb or label image
	# TODO: move to subclass?

	def rgb_image(self, np_array, reverse_colors=True, color_range=(-128, 127)):
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

	def class_image(self, np_array, class_range=None):
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

# Augmentation types and functions

class Augment_Type(dict):
	def __init__(self, outer=None):
		self.outer = outer # Reference to outer class.
		# Set all public methods as mapping/dict.
		#super().__init__({k: v for k, v in self.__class__.__dict__.items() if callable(v) and not k.startswith('_')})
		super().__init__({k: v for k in self.__class__.__dict__ for v in [getattr(self, k)] if not k.startswith('_') and callable(v)})


class Shape(Augment_Type):

	def mirror(self, data, label):
		'''Flips an image of the data and label batches horizontally.'''
		data = data[:,::-1,:]
		label = label[:,::-1,:]

		params = { "flip" : True }

		return data, label, params

	def crop(self, data, label, scale_prob_factor=0.1, replace=0):
		'''Crops an image of the data batch to a smaller rectangle padding the margin with 'replace'. The 'scale_prob_factor' determines the shape of the exponential distribution from which the scaling factor is drawn (higher values mean a smaller rectangle to which the image is cropped).'''

		data = data.copy()

		height, width, channels = data.shape
		rand = self.outer.rand

		# draw random scale values from truncated normal distribution
		scale_x = 1 - rand.random_trunc_exp(1, 0, 1, scale_prob_factor)[0]
		scale_y = 1 - rand.random_trunc_exp(1, 0, 1, scale_prob_factor)[0]

		# draw random x and y positions for the corners of the rectangle that is used to crop the image
		x1 = int(rand.random_unif(1, 0, 1)[0] * (1 - scale_x) * width)
		y1 = int(rand.random_unif(1, 0, 1)[0] * (1 - scale_y) * height)
		x2 = int(x1 + scale_x * width)
		y2 = int(y1 + scale_y * height)

		params = { "factor_x" : scale_x, "factor_y" : scale_y, "box" : [[x1, y1], [x1, y2], [x2, y2], [x2, y1]] }

		# crop image to the calculated rectangle and pad with replacement value
		data[:y1, :, :] = replace
		data[y2:, :, :] = replace
		data[:, :x1, :] = replace
		data[:, x2:, :] = replace

		return data, label, params


	def cut(self, data, label, scale_prob_factor=0.25, replace=0):
		'''Cuts out a rectangle from an image of the data batch and overwrites the values with 'replace'. The 'scale_prob_factor' determines the shape of the exponential distribution from which the scaling factor is drawn (higher values mean a larger rectangle which is replaced).'''

		data = data.copy()

		height, width, channels = data.shape
		rand = self.outer.rand

		# draw random scale values from truncated normal distribution
		scale_x = rand.random_trunc_exp(1, 0, 1, scale_prob_factor)[0]
		scale_y = rand.random_trunc_exp(1, 0, 1, scale_prob_factor)[0]

		# draw random x and y positions for the corners of the rectangle that is replaced
		x1 = int(rand.random_unif(1, 0, 1)[0] * (1 - scale_x) * width)
		y1 = int(rand.random_unif(1, 0, 1)[0] * (1 - scale_y) * height)
		x2 = int(x1 + scale_x * width)
		y2 = int(y1 + scale_y * height)

		params = { "factor_x" : scale_x, "factor_y" : scale_y, "box" : [[x1, y1], [x1, y2], [x2, y2], [x2, y1]] }

		# cut out rectangle and replace values
		# TODO: BUG! Cannot assign to read only memory mapped data (numpy mmap_mode='r')
		data[y1:y2, x1:x2, :] = replace

		return data, label, params


	def scale(self, data, label, scale_prob_factor=0.1):
		'''Rescales an image of the data and label batches. The 'scale_prob_factor' determines the shape of the exponential distribution from which the scaling factor is drawn (higher values mean a smaller square to which the image is zoomed).'''

		data = data.copy()
		label = label.copy()

		height, width, channels = data.shape
		rand = self.outer.rand

		# draw random scale value from truncated normal distribution
		scale = 1 - self.outer.rand.random_trunc_exp(1, 0, 1, scale_prob_factor)[0]

		# draw random x and y positions for the corners of the square that is used to crop the image
		x1 = int(rand.random_unif(1, 0, 1)[0] * (1 - scale) * width)
		y1 = int(rand.random_unif(1, 0, 1)[0] * (1 - scale) * height)
		x2 = int(x1 + scale * width)
		y2 = int(y1 + scale * height)

		# crop image to the calculated square
		data = data[y1:y2, x1:x2, :]
		label = label[y1:y2, x1:x2, :]

		params = { "factor" : scale, "box" : [[x1, y1], [x1, y2], [x2, y2], [x2, y1]] }

		# resize image to the original height and width
		# TODO: OpenCV?
		data = cv.resize(data, dsize=(height, width), interpolation=cv.INTER_LINEAR)
		label = cv.resize(label, dsize=(height, width), interpolation=cv.INTER_NEAREST)[:,:,None]

		return data, label, params


class Color(Augment_Type):

	brightness_deviation = 64
	brightness_max_delta = 128
	contrast_deviation = 0.1
	shift_max_delta = 1
	shift_deviation = 0.1

	def brightness(self, data, label):

		data = data.copy()

		delta = self.outer.rand.random_trunc_norm(1, -self.brightness_max_delta, self.brightness_max_delta, 0, self.brightness_deviation)[0]

		data += delta
		np.clip(data, 0, 512, out=data)
		params = { "delta" : delta }

		return data, label, params

	def contrast(self, data, label):

		gamma = self.outer.rand.random_trunc_norm(1, 0, 2, 1, self.contrast_deviation)[0]
		data = exposure.adjust_gamma(data, gamma)
		params = { "delta" : gamma }

		return data, label, params

	def shift(self, data, label):

		hue = self.outer.rand.random_trunc_norm(1,-self.shift_max_delta, self.shift_max_delta, 0, self.shift_deviation)[0]

		hsv = color.rgb2hsv(data)
		hsv[:, :, 0] = hue
		data = color.hsv2rgb(hsv)

		params = { "delta" : delta }

		return data, label, params
