import numpy as np
from matplotlib import pyplot as plt
from Augmentation import Augmentation

dir = './data/'
data = dir + 'Test_data_CamVid.npy'
label = dir + 'Test_label_CamVid.npy'

if __name__ == '__main__':

	a = Augmentation(aug_type='shape', seed=3)
	data = np.load(data)
	label = np.load(label)

	data_batch = data[0:10,:,:,:]
	label_batch = label[0:10,:,:,:]

	data_batch, label_batch = a.augment_batch(data_batch, label_batch)

	a.rgb_image(data_batch[1,:,:,:])
	plt.show()
