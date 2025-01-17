"""
    search_optimal_aug_params
"""
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


def get_fea(patch):
    fea = np.zeros((256,))
    hist, idx = histogram(patch, nbins=256)
    for hi, idi in zip(hist, idx):
        # print(hi, idi, i, j)
        fea[idi] = hi
    return fea


# def test_mi_of_aug_img():
#     im = cv2.imread("/home1/quanquan/code/landmark/code/tproj/paper_mi/tmp_001.png", cv2.IMREAD_GRAYSCALE)
#     im = cv2.resize(im, (384, 384))
#     lm = [322, 311]
#     ps_half = 32
#     patch = im[lm[0]-ps_half:lm[0]+ps_half, lm[1]-ps_half:lm[1]+ps_half]
#     fea1 = get_fea(patch)
#     entr1 = entropy(fea1)
#     # cv2.imwrite("patch1.jpg", patch)
#     #
#     # fn_aug = transforms.ColorJitter(brightness=0.9)
#     patch_aug = Image.fromarray(patch)
#     patch_aug = F.adjust_brightness(patch_aug, 1.8)
#     patch_aug = F.adjust_contrast(patch_aug, 1.6)
#     patch_aug = np.array(patch_aug)
#     # cv2.imwrite("patch2.jpg", patch_aug)
#     fea2 = get_fea(patch_aug)

#     mi0 = mutual_info_score(fea1.copy(), fea1.copy())
#     mi = mutual_info_score(fea1, fea2)
#     print(mi, mi0)
#     # import ipdb; ipdb.set_trace()


# def _test_mi_fn(patch, br=1.6, ct=1.5):
#     assert len(patch.shape) == 2, f"Got {patch.shape}"
#     assert patch.shape[0] >0 and patch.shape[1] > 0, f"Got {patch.shape}"
#     patch_aug = Image.fromarray(patch)
#     patch_aug = F.adjust_brightness(patch_aug, br)
#     patch_aug = F.adjust_contrast(patch_aug, ct)
#     patch_aug = np.array(patch_aug)
#     assert patch_aug.shape[0] >0 and patch_aug.shape[1] > 0, f"Got {patch_aug.shape}"
#     fea1 = get_fea(patch)
#     fea2 = get_fea(patch_aug)
#     mi0 = mutual_info_score(fea1.copy(), fea1.copy())
#     mi = mutual_info_score(fea1, fea2)
#     entr1 = entropy(fea1)
#     entr2 = entropy(fea2)
#     return mi0, mi, mi/mi0, entr1, entr2


# def search_aug_for_all_patch():
#     from datasets.ceph.ceph_ssl import Test_Cephalometric
#     pth = '/home1/quanquan/datasets/Cephalometric/'
#     testset = Test_Cephalometric(pth, mode="Train")
#     # item, landmark_list, template_patches = testset.__getitem__(0)
#     # data = testset.__getitem__(id_oneshot)
#     landmark_list = []
#     for i in range(len(testset)):
#         landmark = testset.ref_landmarks(i)
#         landmark_list.append(landmark)

#     pth = EX_CONFIG['dataset']['pth']
#     m0_list = []
#     m1_list = []
#     r_list = []
#     en_list = []
#     for i in range(150):
#         landmarks = landmark_list[i]
#         tmp_list0 = []
#         tmp_list1 = []
#         tmp_list2 = []
#         tmp_list3 = []
#         for j, landmark in enumerate(landmarks):
#             print("Processing ", i, j, end="\r")
#             im_pth = tfilename(pth, "RawImage/TrainingData", f"{i+1:03d}.bmp")
#             # print(im_pth)
#             im = cv2.imread(im_pth, cv2.IMREAD_GRAYSCALE)
#             im = cv2.resize(im, (384, 384))
#             lm = landmark
#             ps_half = 32
#             patch = im[max(lm[0]-ps_half, 0):lm[0]+ps_half, max(lm[1]-ps_half, 0):lm[1]+ps_half]
#             mi0, mi, ratio, e1, e2 = _test_mi_fn(patch)
#             tmp_list0.append(mi0)
#             tmp_list1.append(mi)
#             tmp_list2.append(ratio)
#             tmp_list3.append(e1)
#         m0_list.append(tmp_list0)
#         m1_list.append(tmp_list1)
#         r_list.append(tmp_list2)
#         en_list.append(tmp_list3)
#     m0 = np.array(m0_list)
#     m1 = np.array(m1_list)
#     rr = np.array(r_list)
#     en = np.array(en_list)
#     print(m0.mean(), m0.std(), m0.mean(axis=0), m0.std(axis=0))  # 2.781 +- 0.686
#     print(m1.mean(), m1.std(), m1.mean(axis=0), m1.std(axis=0))  # 0.833 +- 0.236
#     print(rr.mean(), rr.std(), rr.mean(axis=0), rr.std(axis=0))  # 0.302 +- 0.07
#     print(en.mean(), en.std(), en.mean(axis=0), en.std(axis=0))  # 4.593 +- 0.506
#     np.save('./cache/mi_ceph_all.npy', [m0, m1, rr, en])
#     import ipdb; ipdb.set_trace()



EX_CONFIG = {
    'dataset': {
        'name': 'Cephalometric',
        'pth': '/home1/quanquan/datasets/Cephalometric/',
        'entr': '/home1/quanquan/datasets/Cephalometric/entr1/train/',
        'prob': '/home1/quanquan/datasets/Cephalometric/prob/train/',
        'n_cls': 19,
    }
}

"""
max mi:  2.9768203987342887
min mi:  0.6626462740608817
"""

def test_all(landmark_list=None, landmark_id=0):
    from torchvision import transforms
    from PIL import Image
    from tutils import tfilename

    pth = EX_CONFIG['dataset']['pth']
    # if landmark_list is None:
    #     from datasets.ceph.ceph_ssl import Test_Cephalometric
    #     testset = Test_Cephalometric(pth, mode="Train")
    #     # item, landmark_list, template_patches = testset.__getitem__(0)
    #     # data = testset.__getitem__(id_oneshot)
    #     landmark_list = []
    #     for i in range(len(testset)):
    #         landmark = testset.ref_landmarks(i)[landmark_id]
    #         landmark_list.append(landmark)

    im_list = []
    fea_list = []
    # landmark_id = 3
    # testset.pth_Image.replace("TrainingData", "NoisyTrainingData25")
    for i in range(150):
        # im_pth = tfilename(pth, "RawImage/NoisyTrainingData75", f"{i+1:03d}.bmp")
        im_pth = tfilename(pth, "RawImage/TrainingData", f"{i+1:03d}.bmp")
        # print(im_pth)
        im = cv2.imread(im_pth, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (384, 384))
        lm = landmark_list[i]
        ps_half = 32
        patch = im[max(lm[0]-ps_half, 0):lm[0]+ps_half, max(lm[1]-ps_half, 0):lm[1]+ps_half]
        if not (patch.shape[0] > 0 and patch.shape[1] > 0):
            import ipdb; ipdb.set_trace()
        # assert patch.shape == (64, 64), f"Got {patch.shape}"
        fea1 = get_fea(patch)
        fea_list.append(fea1)

    fea0 = fea_list[0]
    # mi0 = mutual_info_score(fea0, fea0)
    mi_list = []
    for i in range(150):
        mii = mutual_info_score(fea0, fea_list[i])
        mi_list.append(mii)
    # print(mi_list)
    mi_list = np.array(mi_list)
    print("max mi: ", mi_list.max())
    print("min mi: ",  mi_list.min())
    print("mi_mean: ", mi_list.mean())
    # import ipdb; ipdb.set_trace()
    # draw_bar_auto_split(mi_list)
    # return mi_list.max(), mi_list.min(), mi_list.mean()
    return mi_list


def test_all_sift(lm_idx=0):
    import os
    from einops import rearrange
    config = {
        'dataset_pth': '/home1/quanquan/datasets/Cephalometric/RawImage/TrainingData',
        'sift_save_pth': './runs/sift_landmarks3/RawImage/TrainingData',
        'sift_visuals': './runs/sift_landmarks3/RawImage/TrainingData/visuals/',
        'size': [384, 384],
        'runs_dir': "./runs/",
        'tag': 'sift_landmarks',
    }

    img_pths = [x.path for x in os.scandir(config['dataset_pth']) if x.name.endswith('.bmp')]
    img_pths.sort()
    assert len(img_pths) > 0
    all_landmarks = []
    all_responses = []

    # template
    img = cv2.imread(img_pths[114])
    img = cv2.resize(img, tuple(config['size']))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(gray, None)
    landmark_list = []
    response_list = []
    for kpi in kp0:
        response_list.append(kpi.response)
        landmark_list.append(list(kpi.pt))
    response_list = np.array(response_list)
    # print(response_list)
    rank = np.argsort(response_list)[::-1]
    response_list = response_list[rank]
    # print(response_list)
    lm_list = []
    
    kp_t = np.array(kp0)[rank][lm_idx]
    des_t = des0[rank][lm_idx]
    des_t = rearrange(des_t, "c -> 1 c")
    lm_t = kp_t.pt
    # other imgs
    for img_pth in img_pths:
        img = cv2.imread(img_pth)
        img = cv2.resize(img, tuple(config['size']))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)

        # import ipdb; ipdb.set_trace()
        BF = cv2.BFMatcher()
        matches = BF.knnMatch(des_t, des, k=2)  # k邻近匹配
        # matches[0].distance
        idx = matches[0][0].trainIdx
        pt_i = kp[idx].pt
        # print(pt_i)
        lm_list.append(pt_i)
        # import ipdb; ipdb.set_trace()

    # print("Over!")
    lm_list = np.array(lm_list).round().astype(int)
    return lm_list


def ana_test_all():
    res_list = []
    for i in range(19):
        res = test_all(None, i)
        print("---------------------------")
        print(res)
        print("---------------------------")
        res_list.append(res)
        # import ipdb; ipdb.set_trace()
    res_total = np.concatenate(res_list, axis=0)
    print("res total ", res_total.mean(), res_total.std()) # ceph: mean:1.557; std:0.4429
    # draw_bar_auto_split(res_total, tag=99)
    # np.save(tfilename("./cache/res_total_lm.npy"), res_total)


def search_aug_by_sift():
    res_list = []
    for i in range(1):
        lm = test_all_sift(i)
        # print("lm 114", lm[114])
        res = test_all(lm)
        print(res)
        res_list.append(res)
    res_total = np.concatenate(res_list, axis=0)
    print("res total ", res_total.mean(), res_total.std()) # ceph: mean:1.557; std:0.4429
    np.save(tfilename("./cache/res_total.npy"), res_total)
    # draw_bar_auto_split(res_total, tag=99)


if __name__ == '__main__':
    # test_all(landmark_id=2)
    ana_test_all()
    # search_aug_by_sift()
