"""
    Copied from datasets/ceph/ceph_ssl_entropy.py
"""
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
from tutils import tfilename
from skimage.exposure import histogram
from einops import rearrange
from tqdm import tqdm



class Rectum(data.Dataset):
    """
        Guangdong project: rectum / tumor segmentation
    """
    def __init__(self, pathDataset, mode="Train", size=384, patch_size=192,
                 pre_crop=False, rand_psize=False, ref_landmark=None, prob_map=None,
                 retfunc=1, use_prob=True, ret_name=False, be_consistency=False,
                 cj_brightness=0.15, cj_contrast=0.25, cj_saturation=0., cj_hue=0.,
                 sharpness=0.2, hard_select=False, entr_temp=1,
                 prob_map_dir=None, entr_map_dir=None, num_repeat=10, runs_dir=None):
        assert not (ref_landmark is not None and prob_map is not None), \
            f"Got both ref_landmark: \n{ref_landmark} \nand prob_map: \n{prob_map}"

        self.runs_dir = runs_dir
        self.size = size if isinstance(size, list) else [size, size]
        self.original_size = [2400, 1935]
        self.retfunc = retfunc
        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(pathDataset, '400_junior')
        self.pth_label_senior = os.path.join(pathDataset, '400_senior')
        self.patch_size = patch_size
        self.pre_crop = pre_crop
        self.ref_landmark = ref_landmark
        self.prob_map = prob_map
        self.entr_map_dir = entr_map_dir
        self.prob_map_dir = prob_map_dir
        self.entr_map_dict = None
        self.ret_name = ret_name
        self.mode = mode
        self.base = 16
        self.be_consistency = be_consistency
        self.hard_select = hard_select
        self.entr_temp = entr_temp
        if rand_psize:
            self.rand_psize = rand_psize
            self.patch_size = -1

        self.all_pth_Image = os.path.join(self.pth_Image)
        if mode in ['Oneshot', 'Train']:
            self.pth_Image = self.all_pth_Image[30:]
        elif mode == 'Test':
            self.pth_Image = self.all_pth_Image[:30]
        else:
            raise ValueError

        # ------------------------------------------------
        self.transform = transforms.Compose([transforms.Resize(self.size),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0], [1])])
        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast,
                                   saturation=cj_saturation, hue=cj_hue),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])])
        # for high entropy areas
        self.aug_transform_he = transforms.Compose([
            transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast,
                                   saturation=cj_saturation, hue=cj_hue),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])])
        # for medium entropy areas
        self.aug_transform_me = transforms.Compose([
            transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast,
                                   saturation=cj_saturation, hue=cj_hue),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])])

        self.num_repeat = num_repeat

        self.xy = np.arange(0, self.size[0] * self.size[1])
        self.init_mask()

    def set_rand_psize(self):
        self.patch_size = np.random.randint(6, 8) * 32

    def resize_landmark(self, landmark):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[i] / self.original_size[i])
        return landmark

    # def entr_map_from_image3(self, thresholds=[0,999], inverse=False, temperature=0.5):
    #     """ adjusted entr map, in-threshold: 1, out-threshold: 0 """
    #     print("Get entropy maps! 222 ")
    #     self.entr_map_dict = {}
    #     ratio_list = []
    #     for i in range(150):
    #         name = str(i+1) + ".npy"
    #         path = tfilename(self.entr_map_dir, name)
    #         if not os.path.exists(path):
    #             im_path = os.path.join(self.pth_Image, "{0:03d}".format(i+1) + '.bmp')
    #             entr_map = get_entr_map_from_image(image_path=im_path)
    #             assert entr_map.shape == (384, 384)
    #             # cv2.imwrite(path[:-4] + ".jpg", entr_map, cmap='gray')
    #             np.save(path, entr_map)
    #         else:
    #             entr_map = np.load(path)
    #         # print()
    #         high_entr = np.ones_like(entr_map)
    #         high_entr[np.where(entr_map<thresholds[0])] = 0
    #         high_entr[np.where(entr_map>thresholds[1])] = 0
    #         assert entr_map.shape == (384, 384)
    #         if inverse:
    #             high_entr = high_entr.max() - high_entr
    #         # print("Adjusting porb map, high entr ratio: ", high_entr.sum() / (384*384))
    #         ratio_list.append(high_entr.sum() / (384*384))
    #         # assert entr_map2.sum() == 1, f"Got {entr_map2.shape, entr_map2.sum()}"
    #         self.entr_map_dict["{0:03d}".format(i+1)] = high_entr
    #         if i == 0:
    #             torchvision_save(torch.Tensor(high_entr.copy() / high_entr.max()),
    #                              tfilename(self.runs_dir, name[:-4] + ".png"))
    #     print("avg ratio: ", np.array(ratio_list).mean())
    #     return np.array(ratio_list).mean()

    def select_point_from_entr_map(self, entr_map, size=(192, 192)):
        size_x, size_y = entr_map.shape
        assert size_x == size[0]
        assert size_y == size[1]
        chosen_x1 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
        chosen_y1 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
        chosen_x2 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
        chosen_y2 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
        if (entr_map[chosen_x1, chosen_y1]) * np.random.random() > (entr_map[chosen_x2, chosen_y2]) * np.random.random():
            return chosen_x1, chosen_y1
        else:
            return chosen_x2, chosen_y2

    def select_point_from_probmap_hard(self, prob_map, size=(384,384)):
        prob = rearrange(prob_map, "h w -> (h w)")
        prob = prob / prob.sum()
        loc = np.random.choice(a=self.xy, size=1, replace=True, p=prob)[0]
        return loc // self.size[1], loc % self.size[1]

    def retfunc(self, index):
        """
        New Point Choosing Function with 'PROB MAP'
        """
        np.random.seed()
        item = self.list[index]
        if self.transform != None:
            pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
            item['image'] = self.transform(Image.open(pth_img).convert('RGB'))

        padding = int(0.1 * self.size[0])
        patch_size = self.patch_size
        raw_h, raw_w = self.select_point_from_probmap_hard(self.entr_map_dict[item['ID']], size=self.size)

        b1_left = 0
        b1_top = 0
        b1_right = self.size[0] - patch_size
        b1_bot = self.size[1] - patch_size
        b2_left = raw_w - patch_size + 1
        b2_top = raw_h - patch_size + 1
        b2_right = raw_w
        b2_bot = raw_h
        b_left = max(b1_left, b2_left)
        b_top = max(b1_top, b2_top)
        b_right = min(b1_right, b2_right)
        b_bot = min(b1_bot, b2_bot)
        assert b_left <= b_right
        assert b_top <= b_bot
        left = np.random.randint(b_left, b_right) if b_left < b_right else b_left
        top = np.random.randint(b_top, b_bot) if b_top < b_bot else b_top

        margin_w = left
        margin_h = top
        # print("margin x y", margin_h, margin_w, patch_size)
        # import ipdb; ipdb.set_trace()
        cimg = item['image'][:, margin_h:margin_h + patch_size, margin_w:margin_w + patch_size]
        crop_imgs = augment_patch(cimg, self.aug_transform)
        chosen_w, chosen_h = raw_w - margin_w, raw_h - margin_h

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_h, chosen_w] = 1
        # print("[debug2s] ", crop_imgs.shape, temp.shape)
        temp = cc_augment(torch.cat([crop_imgs, temp], 0))
        crop_imgs = temp[:3]
        # to_PIL(crop_imgs).save('img_after.jpg')
        temp = temp[3]
        # print(chosen_h, chosen_w)
        # chosen_h, chosen_w = temp.argmax() // patch_size, temp.argmax() % patch_size
        chosen_h = torch.div(temp.argmax(), patch_size, rounding_mode="trunc")
        chosen_w = temp.argmax() % patch_size
        return {'raw_imgs': item['image'], 'crop_imgs': crop_imgs, 'raw_loc': torch.LongTensor([raw_h, raw_w]),
                'chosen_loc': torch.LongTensor([chosen_h, chosen_w]), 'ID': item['ID']}

    def init_mask(self):
        # gen mask
        self.Radius = int(max(self.size) * 0.05)
        mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        guassian_mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            for j in range(2*self.Radius):
                distance = np.linalg.norm([i+1 - self.Radius, j+1 - self.Radius])
                if distance < self.Radius:
                    mask[i][j] = 1

        # gen offset
        self.mask = mask
        self.guassian_mask = guassian_mask

        self.offset_x = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        self.offset_y = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            self.offset_x[:, i] = self.Radius - i
            self.offset_y[i, :] = self.Radius - i
        self.offset_x = self.offset_x * self.mask / self.Radius
        self.offset_y = self.offset_y * self.mask / self.Radius

    def gen_mask(self, loc, img_shape=[384, 384]):
        landmark = loc
        y, x = self.size[0], self.size[1]
        mask = torch.zeros((y, x), dtype=torch.float)
        offset_x = torch.zeros((y, x), dtype=torch.float)
        offset_y = torch.zeros((y, x), dtype=torch.float)
        margin_x_left = max(0, landmark[0] - self.Radius)
        margin_x_right = min(x, landmark[0] + self.Radius)
        margin_y_bottom = max(0, landmark[1] - self.Radius)
        margin_y_top = min(y, landmark[1] + self.Radius)
        mask[margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
            self.mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        offset_x[margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
            self.offset_x[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        offset_y[margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] = \
            self.offset_y[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        return mask, offset_y, offset_x

    def __getitem__(self, index):
        return self.retfunc(index)

    def __len__(self):
        return len(self.list)

