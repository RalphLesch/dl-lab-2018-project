import numpy as np
import tensorflow as tf
import Augmentation
from matplotlib import cm, pyplot as plt


mmap_mode='r'

path = 'data/data_CamVidV300/'
data = np.load(path + 'Train_data_CamVid.npy', mmap_mode=mmap_mode)
label = np.load(path + 'Train_label_CamVid.npy', mmap_mode=mmap_mode)

# From https://www.renom.jp/notebooks/tutorial/image_processing/u-net/notebook.html
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

label_name = ['Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 'SignSymbol', 'Fence', 'Car',
              'Pedestrial', 'Bicyclist', 'Unlabelled']
label_colors = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

def visualize(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,11):
        r[temp==l]=label_colors[l,0]
        g[temp==l]=label_colors[l,1]
        b[temp==l]=label_colors[l,2]
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)[:,:,0]
    rgb[:,:,1] = (g/255.0)[:,:,0]
    rgb[:,:,2] = (b/255.0)[:,:,0]
    return rgb
#########################################


with tf.Session() as sess:
	aug = Augmentation.Augmentation(aug_type='all', probability=0.999, seed=1)
	aug.tf_session = sess
	#d, l = aug.augmentations['color'].shift(data[0], label[0])
	#d = tf.image.adjust_contrast(data[0], 0.6)
	#d = sess.run(d)
	#print('equal:', np.array_equal(data[0], d))
	
	'''
	fig = plt.figure()
	fig.add_subplot(1, 2, 1)
	plt.imshow(data[0])
	img = Augmentation.rgb_image(None, data[0].copy())
	fig.add_subplot(1, 2, 2)
	plt.imshow(d)
	img = Augmentation.rgb_image(None, d)
	plt.show()
	'''
	n = 5
	c = 4
	#fig, axs = plt.subplots(nrows=n, ncols=4)
	fig = plt.figure(figsize=(8,10))
	for i in range(0,n):
		print(i)
		p = c*i+1
		fig.add_subplot(n, c, p, title=i).axis('off')
		#plt.imshow(data[i].transpose((1, 2, 0)), cmap='gray')
		aug.rgb_image(data[i].copy())
		p = c*i+2
		fig.add_subplot(n, c, p, title='label').axis('off')
		plt.imshow(visualize(label[i]))
		#aug.class_image(None, label[i])
		
		d, l = aug.augment_img(data[i].copy(), label[i].copy()) # TODO: copy should not be required!
		p = c*i+3
		fig.add_subplot(n, c, p, title='aug').axis('off')
		aug.rgb_image(d)
		p = c*i+4
		fig.add_subplot(n, c, p, title='aug label').axis('off')
		plt.imshow(visualize(l))
		#aug.class_image(None, l)
	fig.tight_layout()
	plt.subplots_adjust(hspace=0.2, wspace=0)
	plt.show()
