

import torchvision.transforms as transforms
import numpy as np
import torch
import os
import torch.utils.data as data
import cv2
from PIL import Image
from datasets.augment import cc_augment
from utils.entropy_loss import get_guassian_heatmaps_from_ref
from skimage.filters.rank import entropy
from scipy.stats import entropy as entropy2
from skimage.morphology import disk
from skimage.exposure import histogram
from einops import rearrange
from tqdm import tqdm


def to_PIL(tensor):
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8).transpose(1, 2, 0))
    return images


def augment_patch(tensor, aug_transform):
    image = to_PIL(tensor)
    # image.save('raw.jpg')
    aug_image = aug_transform(image)
    # aug_image_PIL = to_PIL(aug_image)
    # aug_image_PIL.save('aug.jpg')
    # import ipdb; ipdb.set_trace()
    return aug_image


def im_to_hist(im, patch_size=64, temp=None):
    assert im.shape == (384, 384), f"Got {im.shape}"
    assert len(im.shape) == 2, f"GOt {im.shape}"
    h, w = im.shape
    # mi = np.zeros((h, w))
    ps_h = patch_size // 2
    fea_matrix = np.zeros((256, h, w))
    for i in range(h):
        for j in range(w):
            l1 = max(0, i-ps_h)
            l2 = max(0, j-ps_h)
            patch = im[l1:i+ps_h, l2:j+ps_h]
            hist, idx = histogram(patch, nbins=256)
            # fea = np.zeros((256,))
            for hi, idi in zip(hist, idx):
                # print(hi, idi, i, j)
                fea_matrix[idi, i, j] = hi
    return fea_matrix


def hist_to_entropy(hist):
    c, h, w = hist.shape
    entr_map = np.zeros((h, w))
    for hi in range(h):
        for wi in range(w):
            fea = hist[:, hi, wi]
            # print(fea.shape)
            entr = entropy2(fea)
            if np.isnan(entr) :
                import ipdb; ipdb.set_trace()
            entr_map[hi, wi] = entr
    return entr_map


def get_entr_map_from_image(image_path, size=(384,384)):
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert os.path.exists(image_path), f"Got error path, {image_path}"
    # im = cv2.resize(im, (size[::-1]))
    entr_img = entropy(im, disk(128))
    # entr_img = hist_to_entropy(im_to_hist(im))
    entr_img = cv2.resize(entr_img, (size[::-1]))
    # print("entr_img ", entr_img.shape, size)
    return entr_img