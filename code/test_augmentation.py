import numpy as np
from matplotlib import figure, pyplot as plt
from Augmentation import Augmentation as aug1
from imgaug_augmentation import rgb_image, Augmentation as aug2



mmap_mode='r'

path = 'data/data_CamVidV300/'

data = np.load(path + 'Train_data_CamVid.npy', mmap_mode=mmap_mode)
label = np.load(path + 'Train_label_CamVid.npy', mmap_mode=mmap_mode)

seed=88

img = data[389,:,:,:][None,:,:,:] + 128
segmap = label[389,:,:,:][None,:,:,:] + 1

img -= np.min(img)
img /= np.max(img)
img *= 255

imgs = np.repeat(img, 8, axis=0)
imgs = imgs.reshape((8,300,300,3))
segmaps = np.repeat(segmap, 8, axis=0)
segmaps = segmaps.reshape((8,300,300,1))

a1 = aug1(aug_type="all", seed=seed, probability=0.9)
a2 = aug2(aug_type="all", seed=seed, probability=0.9)

aug_imgs1, aug_segmaps1, infos = a1.augment_batch(imgs,segmaps)
aug_imgs2, aug_segmaps2 = a2.augment_batch(imgs,segmaps)

plot = a2.plot_aug_batch(imgs, segmaps, aug_imgs1, aug_segmaps1, ncols=4)
plt.figure(figsize=(15,28), dpi=300)
rgb_image(plot.astype(np.float32))

plt.axis('off')
# plt.show()
plt.savefig("../poster/figures/aug_hack_example2.png")
plt.savefig("../poster/figures/aug_hack_example2.pdf")
plt.savefig("../poster/figures/aug_hack_example2.svg")

print([", ".join(list(info.keys())) for info in infos])

plot = a2.plot_aug_batch(imgs, segmaps, aug_imgs2, aug_segmaps2, ncols=4)

plt.figure(figsize=(15,28), dpi=300)
rgb_image(plot.astype(np.float32))

plt.axis('off')
# plt.show()
plt.savefig("../poster/figures/imgaug_example2.png")
plt.savefig("../poster/figures/imgaug_example2.pdf")
plt.savefig("../poster/figures/imgaug_example2.svg")
