import numpy as np
from matplotlib import pyplot as plt
from Augmentation import Augmentation as aug1
from imgaug_augmentation import rgb_image, Augmentation as aug2

mmap_mode='r'

path = 'data/data_CamVidV300/'

data = np.load(path + 'Train_data_CamVid.npy', mmap_mode=mmap_mode)
label = np.load(path + 'Train_label_CamVid.npy', mmap_mode=mmap_mode)

seed=16

img = data[0:1,:,:,:] + 128
segmap = label[0:1,:,:,:] + 1

imgs = np.repeat(img, 8, axis=0)
imgs = imgs.reshape((8,300,300,3))
print(imgs.shape)
segmaps = np.repeat(segmap, 8, axis=0)
segmaps = segmaps.reshape((8,300,300,1))

a1 = aug1(aug_type="all", seed=seed, probability=0.9)
a2 = aug2(aug_type="all", seed=seed, probability=0.9)

aug_imgs1, aug_segmaps1, infos = a1.augment_batch(imgs,segmaps)
aug_imgs2, aug_segmaps2 = a2.augment_batch(imgs,segmaps)

#
# rgb_image(imgs[0])
# plt.show()
#
# rgb_image(aug_imgs1[0])
# plt.show()


plot = a2.plot_aug_batch(imgs, segmaps, aug_imgs2, aug_segmaps2, ncols=8)
# plot = a2.plot_aug_batch(imgs, segmaps, aug_imgs2, aug_segmaps2)

print([", ".join(list(info.keys())) for info in infos])

rgb_image(plot.astype(np.float32))
plt.show()
