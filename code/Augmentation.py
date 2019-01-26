import random
import numpy as np
from scipy import special

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

def shape_mirror(self, data, label):
	pass
def shape_crop(self, data, label):
	pass
def shape_cut(self, data, label):
	pass
def shape_scale(self, data, label):
	pass


def color_brightness(self, data, label):
	pass
def color_contrast(self, data, label):
	pass
def color_shift(self, data, label):
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
