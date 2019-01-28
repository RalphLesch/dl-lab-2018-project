import random
import numpy as np
from scipy import special
from matplotlib import cm, pyplot as plt
import tensorflow as tf
from RandomParam import RandomParam


class Augmentation(object):
	#probability = 0
	#random = None
	tf_session = None
	
	def __init__(self, aug_type=None, probability=0.5, seed=None):
		self.augmentations = {'shape': Shape(), 'color': Color()}
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

		for i in range(N):
			x,y = self.augment_img(data[i,:,:,:], label[i,:,:,:])

		return data, label


	def augment_img(self, data, label):
		'''Augment the data and label with the set probability and type.'''

		#print('AUGMENTATION: ', self.type)

		if self._type is None:
			return data, label
		if self._type == 'all':
			# TODO: include no augmentation
			# Probability per typ
			p_type = self.calculate_probability(self.probability, len(self.augmentations))
			for t in self.augmentations:
				data, label = self.random_augmentation_of_type(t, data, label, p_type)
		else:
			data, label = self.random_augmentation_of_type(self._type, data, label, self.probability)

		return data, label

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
		for a_index in augmentations:
			augmentation = t[a_index]
			i = self.rand.random_unif(1, 0, 1)[0]
			if i < p_augment:
				print('augment:', a_index, augmentation)
				data, label = augmentation(self, data, label) # TODO: fix self change!

		return data, label


# Augmentation types and functions

class Augment_Type(dict):
	def __init__(self):
		# Set all public methods as mapping/dict.
		super().__init__({k: v for k, v in self.__class__.__dict__.items() if callable(v) and not k.startswith('_')})
	

class Shape(Augment_Type):

	def mirror(self, data, label):
		'''Flips an image of the data and label batches horizontally.'''

		return np.flip(data, 1), np.flip(label, 1)

	def crop(self, data, label, scale_prob_factor=0.1, replace=-128):
		'''Crops an image of the data batch to a smaller rectangle padding the margin with 'replace'. The 'scale_prob_factor' determines the shape of the exponential distribution from which the scaling factor is drawn (higher values mean a smaller rectangle to which the image is cropped).'''

		height, width, channels = data.shape

		# draw random scale values from truncated normal distribution
		scale_x = 1 - self.rand.random_trunc_exp(1, 0, 1, scale_prob_factor)[0]
		scale_y = 1 - self.rand.random_trunc_exp(1, 0, 1, scale_prob_factor)[0]

		# draw random x and y positions for the corners of the rectangle that is used to crop the image
		x1 = int(self.rand.random_unif(1, 0, 1)[0] * (1 - scale_x) * width)
		y1 = int(self.rand.random_unif(1, 0, 1)[0] * (1 - scale_y) * height)
		x2 = int(x1 + scale_x * width)
		y2 = int(y1 + scale_y * height)

		# crop image to the calculated rectangle and pad with replacement value
		data[:y1, :, :] = replace
		data[y2:, :, :] = replace
		data[:, :x1, :] = replace
		data[:, x2:, :] = replace

		return data, label


	def cut(self, data, label, scale_prob_factor=0.25, replace=-128):
		'''Cuts out a rectangle from an image of the data batch and overwrites the values with 'replace'. The 'scale_prob_factor' determines the shape of the exponential distribution from which the scaling factor is drawn (higher values mean a larger rectangle which is replaced).'''

		height, width, channels = data.shape

		# draw random scale values from truncated normal distribution
		scale_x = self.rand.random_trunc_exp(1, 0, 1, scale_prob_factor)[0]
		scale_y = self.rand.random_trunc_exp(1, 0, 1, scale_prob_factor)[0]

		# draw random x and y positions for the corners of the rectangle that is replaced
		x1 = int(self.rand.random_unif(1, 0, 1)[0] * (1 - scale_x) * width)
		y1 = int(self.rand.random_unif(1, 0, 1)[0] * (1 - scale_y) * height)
		x2 = int(x1 + scale_x * width)
		y2 = int(y1 + scale_y * height)

		# cut out rectangle and replace values
		data[y1:y2, x1:x2, :] = replace

		return data, label


	def scale(self, data, label, scale_prob_factor=0.1):
		'''Rescales an image of the data and label batches. The 'scale_prob_factor' determines the shape of the exponential distribution from which the scaling factor is drawn (higher values mean a smaller square to which the image is zoomed).'''

		height, width, channels = data.shape

		# draw random scale value from truncated normal distribution
		scale = 1 - self.rand.random_trunc_exp(1, 0, 1, scale_prob_factor)[0]

		# draw random x and y positions for the corners of the square that is used to crop the image
		x1 = int(self.rand.random_unif(1, 0, 1)[0] * (1 - scale) * width)
		y1 = int(self.rand.random_unif(1, 0, 1)[0] * (1 - scale) * height)
		x2 = int(x1 + scale * width)
		y2 = int(y1 + scale * height)

		# crop image to the calculated square
		data = data[y1:y2, x1:x2, :]
		label = label[y1:y2, x1:x2, :]

		# resize image to the original height and width
		# TODO: BUG! OpenCV?
		#cv.resize(data, dsize=(height, width), interpolation=cv.INTER_LINEAR)
		#cv.resize(label, dsize=(height, width), interpolation=cv.INTER_NEAREST)

		return data, label


class Color(Augment_Type):
	maxdelta = 0.8
	
	def brightness(self, data, label):
		return data, label
	def contrast(self, data, label):
		return data, label
	def shift(self, data, label):
		return data, label


# ------------------------------------------------------------------------------
# helper functions to plot a rgb or label image
# TODO: move to subclass?

def rgb_image(self, np_array, reverse_colors=True, color_range=(-128, 127)):
	'''Converts a numpy array of shape (height, width, 3) into a RGB image.'''

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

	# normalize classes into range 0 to 1
	np_array -= class_range[0]
	np_array /= n_classes

	# define colormap to distinguish different classes
	colormap = cm.get_cmap(name='nipy_spectral', lut=n_classes).reversed()

	return plt.imshow(np_array, cmap=colormap)
