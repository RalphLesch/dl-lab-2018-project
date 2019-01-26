import random
import numpy as np
from scipy import special, stats
from matplotlib import cm, pyplot as plt
import tensorflow as tf



class Augmentation(object):
	#probability = 0
	#random = None

	def __init__(self, aug_type=None, probability=0.5, seed=None):
		self.type = aug_type
		self.probability = probability
		self.random = random.Random()
		if seed is not None:
			self.random.seed(seed)

	@property
	def type(self):
		return self._type

	@type.setter
	def type(self, aug_type):
		if not (aug_type in [None, 'all', *self.augment_types]):
			raise ValueError("type '{}' is invalid, must be one of {}".format(aug_type, [None, 'all', *self.augment_types]))
		self._type = aug_type

	def augment(self, data, label):
		'''Augment the data and label (in place) with the set probability and type.'''

		#print('AUGMENTATION: ', self.type)

		if self._type is None:
			return
		if self._type == 'all':
			# TODO: include no augmentation
			# Probability per type
			p_type = self.calculate_probability(self.probability, len(self.augment_types))
			for t in self.augment_types:
				self.random_augmentation_of_type(t, data, label, p_type)
		else:
			self.random_augmentation_of_type(self._type, data, label, self.probability)

		#return data, label

	def calculate_probability(self, base_probability, n_elements):
		''' Calculate probablility for each augmentation (or type).'''

		# calculate coefficients of the polynomial (identical to binomial coefficients) inferred from the inclusion–exclusion principle (P(A or B) = P(B) + P(A) - P(A and B) and P(A) = P(B))
		coefs = [-base_probability] + [special.binom(n_elements, k) * (-1)**(k+1) for k in range(1, n+1)]
		coefs.reverse()

		# calculate real valued roots of the polynomial between 0 and 1 (representing a the probability of P(A) = P(B))
		roots = np.roots(coefs)
		roots = np.real(roots[np.isreal(roots)])
		roots = roots[(roots > 0.0) & (roots < 1.0)]

		return roots[0]

	def random_augmentation_of_type(self, aug_type, data, label, probability):
		''' Apply random augmentation of the given type (see augment_types) to data and label
			(changed in place), with the given base probability.
		'''
		t = self.augment_types[aug_type]
		# Probability for each augmentation.
		p_augment = self.calculate_probability(probability, len(t))
		augmentations = list(t.keys())
		self.random.shuffle(augmentations)
		for a_index in augmentations:
			augmentation = t[a_index]
			i = self.random.randint(0, 1)
			if i < p_augment:
				#data, label = augmentation(data, label)
				print('augment: ' + a_index)

	#class augment_types(object):
	#	class shape:
	#		def mirror(self, data, label):
	#			pass
	#		def crop(self, data, label):
	#			pass
	#		def cut(self, data, label):
	#			pass
	#		def scale(self, data, label):
	#			pass
	#	class color:
	#		def brightness(self, data, label):
	#			pass
	#		def contrast(self, data, label):
	#			pass
	#		def shift(self, data, label):
	#			pass


# TODO: move to sub-class?

def shape_mirror(self, data, label, index=None):
	pass
def shape_crop(self, data, label, index=None):
	pass
def shape_cut(self, data, label, index=None):
	pass
def shape_scale(self, data, label, index=None, scale_prob_factor=0.1):
	'''Scales all images of the batch with a given index. If scale_factors and position are not specified random parameter values are drawn.'''

	N, height, width, channels = data.get_shape().as_list()

	# if index is not specified all images of the batch are rescaled
	if index is None:
		index = list(range(N))

	# draw random scale values from truncated normal distribution
	scales = 1 - random_trunc_exp(len(index), 0, 1, scale_prob_factor)
	# draw random x and y positions of the rectangle from uniform distribution
	positions = np.column_stack([random_unif(len(index), 0, 1), random_unif(len(index), 0, 1)])

	# calculate corners of the rectangle that is used to rescale the image
	y1 = (1 - scales) * positions[:,0]
	y2 = 1 - (1 - scales) * (1 - positions[:,0])
	x1 = (1 - scales) * positions[:,1]
	x2 = 1 - (1 - scales) * (1 - positions[:,1])
	boxes = np.column_stack([y1, x1, y2, x2])

	# crop and resize the image by the calculated coordinates
	data = tf.image.crop_and_resize(data, boxes, index, [height, width])
	labels = tf.image.crop_and_resize(labels, boxes, index, [height, width])

	return data, labels, (scales, boxes)


def color_brightness(self, data, label, index=None):
	pass
def color_contrast(self, data, label, index=None):
	pass
def color_shift(self, data, label, index=None):
	pass

Augmentation.augment_types = {
	'shape': {
		'mirror': shape_mirror
		,'crop': shape_crop
		,'cut': shape_cut
		,'scale': shape_scale
	}
	,'color': {
		'brightness': color_brightness
		,'contrast': color_contrast
		,'shift': color_shift
	}
}


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

# ------------------------------------------------------------------------------
# helper functions to plot and draw samples from different distributions
# TODO: move to subclass?

def density_trunc_exp(self, lower, upper, scale):
	'''Samples x and y values in order to plot the truncated exponential distribution with the given parameters.'''

	b = (upper-lower) / scale
	x = np.linspace(lower, upper, 100)
	y = stats.truncexpon(b, loc=lower, scale=scale).pdf(x)
	return x, y

def random_trunc_exp(self, N, lower, upper, scale):
	'''Draws N random values from the truncated exponential distribution.'''

	b = (upper-lower) / scale
	return stats.truncexpon(b, loc=lower, scale=scale).rvs(N)

def density_trunc_norm(self, lower, upper, mean, std):
	'''Samples x and y values in order to plot the truncated normal distribution with the given parameters.'''

	a = (lower - mean) / std
	b = (upper - mean) / std
	x = np.linspace(lower, upper, 100)
	y = stats.truncnorm(a, b, loc=mean, scale=std).pdf(x)
	return x, y

def random_trunc_norm(self, N, lower, upper, mean, std):
	'''Draws N random values from the truncated exponential distribution.'''

	a = (lower - mean) / std
	b = (upper - mean) / std
	return stats.truncnorm(a, b, loc=mean, scale=std).rvs(N)

def random_unif(self, N, lower, upper):
	'''Draws N random values between lower and upper from the uniform distribution.'''

	return stats.uniform(loc=lower, scale=upper-lower).rvs(N)
