from .hand_ssl import HandXray as basic_dataset
from .hand_ssl import get_csv_content
import pandas as pd
from tutils import tfilename
import torchvision.transforms as transforms
import numpy as np
import os


HARD_LIST = [  0,   1,   2,   3,   4,   5,   6,   7,   9,  11,  13,  14,  15,
    16,  24,  31,  32,  33,  35,  36,  48,  51,  54,  61,  64,  65,
    67,  70,  72,  73,  75,  81,  82,  84,  86,  90,  93,  95,  99,
    100, 108, 109, 121, 123, 126, 127, 129, 131, 137, 140, 141, 147,
    152, 153, 155, 159, 165, 169, 179, 181, 190, 191, 198, 211, 216,
    234, 236, 243, 248, 258, 261, 262, 279, 291, 292, 297, 306, 309,
    316, 319, 320, 328, 356, 360, 378, 381, 390, 391, 392, 393, 394,
    395, 402, 403, 404, 405, 406, 408, 409, 420, 424, 426, 428, 447,
    472, 473, 479, 487, 488, 495, 532, 543, 550, 556, 583, 592]


def reverse_filter(ll, indices):
    new_ll = []
    for i in range(len(ll)):
        if i in indices:
            continue
        new_ll.append(ll[i])
    return new_ll


class HandXray(basic_dataset):
    # '/home/quanquan/hand/hand/histo-norm/'
    def __init__(self, pathDataset='/home1/quanquan/hand/hand/',
                 split="Hard", size=[384, 384],
                 extra_aug_loss=False, patch_size=192, rand_psize=False,
                 cj_brightness=0.25, cj_contrast=0.15,
                 retfunc=1, num_repeat=1):
        """
        extra_aug_loss: return an additional image for image augmentation consistence loss
        """
        self.size = size
        self.extra_aug_loss = extra_aug_loss  # consistency loss
        self.pth_Image = tfilename(pathDataset, "jpg")
        self.patch_size = patch_size
        self.num_repeat = num_repeat

        self.list = [x.path for x in os.scandir(self.pth_Image) if x.name.endswith(".jpg")]
        self.list.sort()
        # print(self.list)
        label_path = tfilename(pathDataset, "all.csv")
        self.landmarks = get_csv_content(label_path)
        self.test_list = self.list[:300]
        self.landmarks_test = self.landmarks[:300]
        self.train_list = self.list[300:]
        self.landmarks_train = self.landmarks[300:]
        
        if split == "HARD":
            self.train_list = [self.train_list[i] for i in HARD_LIST]        
            self.landmarks_train = [self.landmarks_train[i] for i in HARD_LIST]        
        elif split == "EASY":
            self.train_list = reverse_filter(self.train_list, HARD_LIST)
            self.landmarks_train = reverse_filter(self.landmarks_train, HARD_LIST)        

        if split in ["HARD", "EASY"]:
            self.istrain = True
        elif split in ["Test1", "Test"]:
            self.istrain = False
        else:
            raise NotImplementedError

        normalize = transforms.Normalize([0.], [1.])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)

        self.transform_resize = transforms.Resize(self.size)
        self.transform_tensor = transforms.ToTensor()

        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=cj_brightness, contrast=cj_contrast),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ])

        self.retfunc = retfunc
        self.mode = split
        self.base = 16

        if split in ["HARD", "EASY"]:
            self.loading_list = self.train_list
            self.real_len = len(self.loading_list)
        else:
            raise NotImplementedError
