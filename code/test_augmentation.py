import numpy as np
from matplotlib import pyplot as plt
from Augmentation import Augmentation as aug1
from imgaug_augmentation import rgb_image, Augmentation as aug2

mmap_mode='r'

path = 'data/data_CamVidV300/'

data = np.load(path + 'Train_data_CamVid.npy', mmap_mode=mmap_mode)
label = np.load(path + 'Train_label_CamVid.npy', mmap_mode=mmap_mode)

imgs = data[0:24,:,:,:] + 128
segmaps = label[0:24,:,:,:] + 1

a1 = aug1(aug_type="all", seed=14, probability=0.9)
a2 = aug2(aug_type="all", seed=14, probability=0.9)


aug_imgs1, aug_segmaps1, info = a1.augment_batch(imgs,segmaps)
aug_imgs2, aug_segmaps2 = a2.augment_batch(imgs,segmaps)

print(info)

rgb_image(imgs[0])
plt.show()

rgb_image(aug_imgs1[0])
plt.show()


plot = a2.plot_aug_batch(imgs, segmaps, aug_imgs1, aug_segmaps1)
# plot = a2.plot_aug_batch(imgs, segmaps, aug_imgs2, aug_segmaps2)

rgb_image(plot.astype(np.float32))
plt.show()
