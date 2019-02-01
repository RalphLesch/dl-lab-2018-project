import numpy as np
from matplotlib import figure, pyplot as plt
from Augmentation import Augmentation as aug1
from imgaug_augmentation import Augmentation as aug2
from plots import *
import imgaug as ia
from imgaug import augmenters as iaa

mmap_mode='r'

path = 'data/data_CamVidV300/'

data = np.load(path + 'Train_data_CamVid.npy', mmap_mode=mmap_mode)
label = np.load(path + 'Train_label_CamVid.npy', mmap_mode=mmap_mode)

seed=88

img = data[389,:,:,:][None,:,:,:] + 128
segmap = label[389,:,:,:][None,:,:,:] + 1

# img -= np.min(img)
# img /= np.max(img)
# img *= 255
#
# imgs = np.repeat(img, 8, axis=0)
# imgs = imgs.reshape((8,300,300,3))
# segmaps = np.repeat(segmap, 8, axis=0)
# segmaps = segmaps.reshape((8,300,300,1))
#
# a1 = aug1(aug_type="all", seed=seed, probability=0.9)
# a2 = aug2(aug_type="all", seed=seed, probability=0.9)
#
# aug_imgs1, aug_segmaps1, infos = a1.augment_batch(imgs,segmaps)
# aug_imgs2, aug_segmaps2 = a2.augment_batch(imgs,segmaps)
#
# plot = plot_aug_batch(imgs, segmaps, aug_imgs1, aug_segmaps1, ncols=4)
# plt.figure(figsize=(15,28), dpi=300)
# rgb_image(plot.astype(np.float32))
#
# plt.axis('off')
# plt.show()
# # plt.savefig("../poster/figures/aug_hack_example2.png")
# # plt.savefig("../poster/figures/aug_hack_example2.pdf")
# # plt.savefig("../poster/figures/aug_hack_example2.svg")
#
# print([", ".join(list(info.keys())) for info in infos])
#
# plot = plot_aug_batch(imgs, segmaps, aug_imgs2, aug_segmaps2, ncols=4)
#
# plt.figure(figsize=(15,28), dpi=300)
# rgb_image(plot.astype(np.float32))
#
# plt.axis('off')
# plt.show()
# # plt.savefig("../poster/figures/imgaug_example2.png")
# # plt.savefig("../poster/figures/imgaug_example2.pdf")
# # plt.savefig("../poster/figures/imgaug_example2.svg")
#

# ------------------------------------------------------------------------------

sequences = [

	iaa.Sequential([
		iaa.Fliplr(1),
		iaa.CoarseDropout(0.1, size_percent=0.02)
	], random_order=True),

	iaa.Sequential([
		iaa.Crop(px=54),
		iaa.Pad(px=16, pad_mode=ia.ALL, pad_cval=(0, 128)),
	], random_order=False),

	iaa.Sequential([
		iaa.GammaContrast(0.92),
		iaa.AdditiveGaussianNoise(loc=0, scale=(0.02*255))
	], random_order=True),

	iaa.Sequential([
		iaa.SaltAndPepper(0.005),
		iaa.GammaContrast((1.05), per_channel=True)
	], random_order=True),

	iaa.Sequential([
		iaa.CoarseDropout(0.1, size_percent=0.02),
		iaa.GammaContrast((0.95), per_channel=True),
		iaa.Fliplr(1),
		iaa.Crop(px=34),
	], random_order=True),

	iaa.Sequential([
		iaa.GammaContrast((1.19)),
		iaa.SaltAndPepper(0.005),
		iaa.Pad(px=20, pad_mode=ia.ALL, pad_cval=(0, 128)),
		iaa.AdditiveGaussianNoise(loc=0, scale=(0.015*255))
	], random_order=True)
]

imgs = np.repeat(img, len(sequences), axis=0)
imgs = imgs.reshape((len(sequences),300,300,3))
segmaps = np.repeat(segmap, len(sequences), axis=0)
segmaps = segmaps.reshape((len(sequences),300,300,1))

def augment(img, segmap, seq):

	seq_det = seq.to_deterministic()
	aug_img = seq_det.augment_images(img)
	aug_segmap = seq_det.augment_segmentation_maps([segmap])[0]

	return aug_img, aug_segmap

segmap = aug2.np2segmap(segmap[0])


# for i in range(420,440):
# 	print(i)
ia.seed(439)

aug_imgs2 = np.zeros((len(sequences), 300, 300, 3))
aug_segmaps2 = np.zeros((len(sequences), 300, 300, 1))

for i, seq in enumerate(sequences):
	aug_img, aug_segmap = augment(img, segmap, seq)
	aug_imgs2[i] = aug2.img2np(aug_img)
	aug_segmaps2[i] = aug2.segmap2np(aug_segmap)

plot = plot_aug_batch(imgs[:6], segmaps[:6], aug_imgs2, aug_segmaps2, ncols=4)

plt.figure(figsize=(16,24), dpi=300)
rgb_image(plot.astype(np.float32))

plt.savefig("../poster/figures/imgaug_example2.png")
plt.savefig("../poster/figures/imgaug_example2.pdf")
plt.savefig("../poster/figures/imgaug_example2.svg")

plt.axis('off')
plt.show()
