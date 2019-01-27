# ------------------------------------------------------------------------------
# helper class to plot and draw samples from different random distributions

import numpy as np
from scipy import stats

class RandomParam(object):

	def __init__(self, seed=None):
		self.random = np.random.RandomState(seed)

	def density_trunc_exp(self, lower, upper, scale):
		'''Samples x and y values in order to plot the truncated exponential distribution with the given parameters.'''

		b = (upper-lower) / scale
		x = np.linspace(lower, upper, 100)
		y = stats.truncexpon(b, loc=lower, scale=scale).pdf(x)
		return x, y

	def random_trunc_exp(self, N, lower, upper, scale):
		'''Draws N random values from the truncated exponential distribution.'''

		b = (upper-lower) / scale
		return stats.truncexpon(b, loc=lower, scale=scale).rvs(N, random_state=self.random)

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
		return stats.truncnorm(a, b, loc=mean, scale=std).rvs(N, random_state=self.random)

	def random_unif(self, N, lower, upper):
		'''Draws N random values between lower and upper from the uniform distribution.'''

		return stats.uniform(loc=lower, scale=upper-lower).rvs(N, random_state=self.random)

	def shuffle(self, x):
		'''Shuffles a given array x in place.'''

		self.random.shuffle(x)
