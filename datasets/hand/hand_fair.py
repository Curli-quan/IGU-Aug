from .hand_ssl import HandXray as basic_dataset
from .hand_ssl import get_csv_content
import pandas as pd
from tutils import tfilename
import torchvision.transforms as transforms
import numpy as np
import os


def find_male_list(all_list, pd, gender="M"):
    assert gender in ["M", "F"]
    all_list = [os.path.split(item)[-1].replace(".jpg","") for item in all_list]
    # data = pd[pd['id'].isin(all_list)]
    check_list = []
    for i, item in enumerate(all_list):
        agender = pd[pd['id']==int(item)]['gender'].to_numpy()[0]
        if agender==gender:
            check_list.append(i)
    return check_list

class HandXray(basic_dataset):
    # '/home/quanquan/hand/hand/histo-norm/'
    def __init__(self, pathDataset='/home1/quanquan/hand/hand/',
                 gender="M",
                 split="Train", size=[384, 384],
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
        
        self.personal_info = pd.read_csv("/home1/quanquan/code/landmark/code/tproj/datasets/hand/hand_sensitive_info.csv")
        gender_list = find_male_list(self.train_list, self.personal_info, gender=gender)
        self.train_list = [self.train_list[i] for i in gender_list]        
        self.landmarks_train = [self.landmarks_train[i] for i in gender_list]        
        
        gender_list = find_male_list(self.test_list, self.personal_info, gender=gender)
        self.test_list = [self.test_list[i] for i in gender_list]        
        self.landmarks_test = [self.landmarks_test[i] for i in gender_list]   

        if split in ["Oneshot", "Train"]:
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

        if split in ['Train', 'Oneshot']:
            self.loading_list = self.train_list
            self.real_len = len(self.loading_list)
        else:
            raise NotImplementedError
