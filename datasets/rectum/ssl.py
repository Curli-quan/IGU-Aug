"""
    from cas-qs/datasets/data_loader_gp.py
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
import json
import glob


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


class RectumDataset(data.Dataset):
    def __init__(self, pathDataset="/home1/zhhli/sagjpg/", mode="Train", size=384, patch_size=192,
                 pre_crop=False, rand_psize=False, ref_landmark=None, prob_map=None,
                 retfunc=1, use_prob=False, ret_name=False, be_consistency=False,
                 cj_brightness=0.15, cj_contrast=0.25, cj_saturation=0., cj_hue=0.,
                 sharpness=0.2, num_repeat=10):
        assert not (ref_landmark is not None and prob_map is not None), \
            f"Got both ref_landmark: \n{ref_landmark} \nand prob_map: \n{prob_map}"

        self.size = size if isinstance(size, list) else [size, size]
        self.original_size = [512, 512]
        self.retfunc = retfunc
        self.patch_size = patch_size
        self.pre_crop = pre_crop
        self.ref_landmark = ref_landmark
        self.prob_map = prob_map
        self.ret_name = ret_name
        if rand_psize:
            self.rand_psize = rand_psize
            self.patch_size = -1

        self.list = glob.glob(os.path.join(pathDataset, "*.jpg"))

        if mode in ['Oneshot', 'Train']:
            self.pth_Image = self.list[30:]
        elif mode == 'Test':
            self.pth_Image = self.list[:30]
        else:
            raise ValueError

        self.pre_trans = transforms.Compose([transforms.RandomCrop((int(512 * 0.9), int(512 * 0.9)))])

        self.transform = transforms.Compose([transforms.Resize(self.size),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0], [1]),])
        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast,
                                   saturation=cj_saturation, hue=cj_hue),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])


        self.num_repeat = num_repeat
        self.mode = mode
        self.base = 16
        self.be_consistency = be_consistency

        self.init_mask()

    def set_rand_psize(self):
        self.patch_size = np.random.randint(6, 8) * 32

    def resize_landmark(self, landmark):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[i] / self.original_size[i])
        return landmark

    def select_point_from_prob_map(self, prob_map, size=(192, 192)):
        size_x, size_y = prob_map.shape
        assert size_x == size[0], f"Got {size_x}, {size}"
        assert size_y == size[1], f"Got {size_y}, {size}"
        chosen_x1 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
        chosen_x2 = np.random.randint(int(0.1 * size_x), int(0.9 * size_x))
        chosen_y1 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
        chosen_y2 = np.random.randint(int(0.1 * size_y), int(0.9 * size_y))
        if prob_map[chosen_x1, chosen_y1] * np.random.random() > prob_map[chosen_x2, chosen_y2] * np.random.random():
            return chosen_x1, chosen_y1
        else:
            return chosen_x2, chosen_y2

    def retfunc1(self, index):
        """
        New Point Choosing Function without prob map
        """
        np.random.seed()
        item = {}
        if self.transform != None:
            pth_img = os.path.join(self.list[index])
            item['image'] = self.transform(Image.open(pth_img).convert('RGB'))
        pad_scale = 0.03
        patch_size = self.patch_size
        # raw_w, raw_h = self.select_point_from_prob_map(self.prob_map, size=self.size)
        raw_w = np.random.randint(int(pad_scale * self.size[1]), int((1 - pad_scale) * self.size[1]))
        raw_h = np.random.randint(int(pad_scale * self.size[0]), int((1 - pad_scale) * self.size[0]))
        # print(raw_h, raw_w)
        b1_left = 0
        b1_top = 0
        b1_right = self.size[1] - patch_size
        b1_bot = self.size[0] - patch_size
        b2_left = raw_w - patch_size + 1
        b2_top = raw_h - patch_size + 1
        b2_right = raw_w
        b2_bot = raw_h
        b_left = max(b1_left, b2_left)
        b_top = max(b1_top, b2_top)
        b_right = min(b1_right, b2_right)
        b_bot = min(b1_bot, b2_bot)
        left = np.random.randint(b_left, b_right)
        top = np.random.randint(b_top, b_bot)

        # assert top
        margin_w = left
        margin_h = top
        
        cimg = item['image'][:, margin_h:margin_h + patch_size, margin_w:margin_w + patch_size]
        # print(cimg.shape)
        assert cimg.shape[1] == patch_size and cimg.shape[2] == patch_size, f"Got {cimg.shape}"

        crop_imgs = augment_patch(cimg, self.aug_transform)
        chosen_w, chosen_h = raw_w - margin_w, raw_h - margin_h

        temp = torch.zeros([1, patch_size, patch_size])
        temp[:, chosen_h, chosen_w] = 1
        # print(crop_imgs.shape, temp.shape)
        temp = cc_augment(torch.cat([crop_imgs, temp], 0))
        crop_imgs = temp[:3]
        temp = temp[3]
        # chosen_h, chosen_w = temp.argmax() // patch_size, temp.argmax() % patch_size
        chosen_h = torch.div(temp.argmax(), patch_size, rounding_mode="trunc")
        chosen_w = temp.argmax() % patch_size

        return {'raw_imgs': item['image'],
                'crop_imgs': crop_imgs,
                'raw_loc': torch.LongTensor([raw_h, raw_w]),
                'chosen_loc': torch.LongTensor([chosen_h, chosen_w]),
                "index": index, "path": self.list[index]}

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
        index = index % len(self.list)
        if self.retfunc == 1:
            return self.retfunc1(index)
        elif self.retfunc == 2:
            return self.retfunc2(index)
        elif self.retfunc == 3:
            return self.retfunc3(index)
        elif self.retfunc == 0:
            return self.retfunc_old(index)
        else:
            raise ValueError

    def __len__(self):
        return len(self.list) * self.num_repeat


def test(logger, config, *args, **kwargs):
    dataset = RectumDataset(mode="Oneshot")
    item = dataset.__getitem__(0)
    import ipdb;     ipdb.set_trace()



if __name__ == "__main__":
    from torchvision.utils import save_image
    from tutils.new.manager import trans_init, trans_args
    args = trans_args()
    logger, config = trans_init(args, file=__file__)
    test(logger, config)
    # test_prob_map()