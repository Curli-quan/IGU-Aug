import cv2
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
# from scipy.misc import imread
import numpy as np
import cv2
from skimage.exposure import histogram
from scipy.stats import entropy
import matplotlib
from tutils import tfilename
from PIL import Image
import torchvision.transforms.functional as F
from scipy.stats import norm
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from skimage.filters.rank import entropy
# from scipy.stats import entropy
from skimage.morphology import disk
from tqdm import tqdm
from torchvision.utils import save_image
import torch

mi_mean = [1.1777022788]
mi_std = [0.432]

def get_fea(patch):
    fea = np.zeros((256,))
    hist, idx = histogram(patch, nbins=256)
    for hi, idi in zip(hist, idx):
        # print(hi, idi, i, j)
        fea[idi] = hi
    return fea

def _tmp_fn2(landmark_id, alpha=1):
    """ augment and crop patch (different from _tmp_fn() ) """
    from datasets.ceph.ceph_ssl import Test_Cephalometric
    pth = '/home1/quanquan/datasets/Cephalometric/'
    testset = Test_Cephalometric(pth, mode="Train")
    # Process: lm 0, 3, 7
    lms = testset.ref_landmarks(114)   

    dirname = '/home1/quanquan/datasets/Cephalometric/RawImage/TrainingData/'

    params_list = []
    for i in range(150):
        i += 1
        # i = 115
        # landmark_id = 3
        
        im_name = f'{i:03d}.bmp'
        im = cv2.imread(dirname + im_name, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (384, 384))

        lm = lms[landmark_id]  # [168, 163]
        print()
        ps_half = 32
        patch = im[lm[1] - ps_half:lm[1] + ps_half, lm[0] - ps_half:lm[0] + ps_half]
        # save image patch
        # save_image(torch.Tensor(patch) / 255, f"paper_mi/saved_patches/im{i:03d}_lm{landmark_id}_patch.png")
        # continue
        fea1 = get_fea(patch)
        im = Image.fromarray(im)

        # print("=============================")
        cj_brightness = 1.
        cj_contrast = 1.
        params1 = [0, 0]
        for i in range(100):
            im_aug = F.adjust_brightness(im, cj_brightness)
            im_aug = F.adjust_contrast(im_aug, cj_contrast)
            im_aug = np.array(im_aug)
            # cv2.imwrite("patch2.jpg", patch_aug)
            patch_aug = im_aug[lm[1] - ps_half:lm[1] + ps_half, lm[0] - ps_half:lm[0] + ps_half]
            fea2 = get_fea(patch_aug)
            # mi0 = mutual_info_score(fea1.copy(), fea1.copy())
            mi = mutual_info_score(fea1, fea2)
            print(f"br: {cj_brightness}, ct: {cj_contrast}, mi:", mi, mi_mean[landmark_id] , mi_std[landmark_id]) # mi0
            cj_brightness += 0.03
            cj_contrast += 0.04
            if mi < mi_mean[landmark_id] - mi_std[landmark_id]:  # - mi_std[landmark_id]:
                break
            params1 = [cj_brightness, cj_contrast]
        print("=============================")

        cj_brightness = 1
        cj_contrast = 1
        params2 = [0, 0]
        for i in range(100):
            im_aug = F.adjust_brightness(im, cj_brightness)
            im_aug = F.adjust_contrast(im_aug, cj_contrast)
            im_aug = np.array(im_aug)
            # cv2.imwrite("patch2.jpg", patch_aug)
            patch_aug = im_aug[lm[1] - ps_half:lm[1] + ps_half, lm[0] - ps_half:lm[0] + ps_half]
            fea2 = get_fea(patch_aug)

            # mi0 = mutual_info_score(fea1.copy(), fea1.copy())
            mi = mutual_info_score(fea1, fea2)
            print(f"br: {cj_brightness}, ct: {cj_contrast}, mi:", mi, mi_mean[landmark_id] , mi_std[landmark_id])  # mi0
            cj_brightness -= 0.03
            cj_contrast -= 0.04
            if mi < mi_mean[landmark_id] - mi_std[landmark_id]:
                break
            params2 = [cj_brightness, cj_contrast]
        # print("=============================")
        params_list.append(params1 + params2)
        # import ipdb; ipdb.set_trace()

    params_list = np.array(params_list)
    mean_values = params_list.mean(axis=0)
    return mean_values

dd = _tmp_fn2(0,1)
print(dd)