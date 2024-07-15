import torchvision.transforms as transforms
import numpy as np
import torch
import os
import torch.utils.data as data
import cv2
from PIL import Image
from datasets.augment import cc_augment
from utils.entropy_loss import get_guassian_heatmaps_from_ref


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


class Cephalometric(data.Dataset):
    def __init__(self, pathDataset='/home1/quanquan/datasets/Cephalometric/', split="Oneshot", size=384, patch_size=192,
                 pre_crop=False, rand_psize=False, ref_landmark=None, prob_map=None,
                 retfunc=1, use_prob=False, ret_name=False, be_consistency=False,
                 num_repeat=10):
        assert not (ref_landmark is not None and prob_map is not None), \
            f"Got both ref_landmark: \n{ref_landmark} \nand prob_map: \n{prob_map}"

        self.size = size if isinstance(size, list) else [size, size]
        self.original_size = [2400, 1935]
        self.retfunc = retfunc
        if retfunc > 0:
            print("Using new ret Function!")
            self.new_ret = True
        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_label_junior = os.path.join(pathDataset, '400_junior')
        self.pth_label_senior = os.path.join(pathDataset, '400_senior')
        self.patch_size = patch_size
        self.pre_crop = pre_crop
        self.ref_landmark = ref_landmark
        self.prob_map = prob_map
        self.ret_name = ret_name
        if rand_psize:
            self.rand_psize = rand_psize
            self.patch_size = -1

        self.list = list()

        if split in ['Oneshot', 'Train']:
            self.pth_Image = os.path.join(self.pth_Image, 'TrainingData')
            start = 1
            end = 150
        elif split == 'Test1':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 300
        elif split == 'Test2':
            self.pth_Image = os.path.join(self.pth_Image, 'Test2Data')
            start = 301
            end = 400
        elif split == 'Test1+2':
            self.pth_Image = os.path.join(self.pth_Image, 'Test1Data')
            start = 151
            end = 400
        else:
            raise ValueError

        self.pre_trans = transforms.Compose([transforms.RandomCrop((int(2400 * 0.8), int(1935 * 0.8)))])
        normalize = transforms.Normalize([0], [1])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        # transforms.ColorJitter(brightness=0.15, contrast=0.25),
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)

        transform_list = [
            transforms.ColorJitter(brightness=0.15, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ]
        self.aug_transform = transforms.Compose(transform_list)

        for i in range(start, end + 1):
            self.list.append({'ID': "{0:03d}".format(i)})

        num_repeat = num_repeat
        if split == 'Train' or split == 'Oneshot':
            temp = self.list.copy()
            for _ in range(num_repeat):
                self.list.extend(temp)

        self.split = split
        self.base = 16
        self.be_consistency = be_consistency

    def resize_landmark(self, landmark):
        for i in range(len(landmark)):
            landmark[i] = int(landmark[i] * self.size[i] / self.original_size[i])
        return landmark
    
    def retfunc1(self, index):
        """
        New Point Choosing Function without prob map
        """
        # np.random.seed()
        item = self.list[index]
        pth_img = os.path.join(self.pth_Image, item['ID'] + '.bmp')
        item['image'] = Image.open(pth_img).convert('RGB')
        if self.transform != None:
            item['image'] = self.transform(item['image'])

        resize = transforms.Resize(self.size)        
        item['image'] = resize(item['image'])

        sift_dir = '/home1/quanquan/code/landmark/code/runs/sift/sift/s1/lm/'
        sift_response_pth = sift_dir + f"sift_responses_{item['ID']}.npy"
        sift_landmark_pth = sift_dir + f"sift_landmarks_{item['ID']}.npy"
        sift_descript_pth = sift_dir + f"sift_descriptor_{item['ID']}.npy"
        sift_response = np.load(sift_response_pth)
        sift_landmark = np.load(sift_landmark_pth)
        sift_descript = np.load(sift_descript_pth)

        sift_landmark = torch.Tensor(sift_landmark).round().long() # // 2
        # w, h = item['image'].shape[1] // 2, item['image'].shape[2] // 2
        # print("shape: w, h ", w, h )
        # sift_descript = torch.Tensor(sift_descript)
        # landmark_map = torch.zeros((w, h))
        # response_map = torch.zeros((w, h))
        # descript_map = torch.zeros((128, w, h))
        # for i, lm in enumerate(sift_landmark):
        #     descript_map[:, lm[0], lm[1]] = sift_descript[i]
        #     response_map[lm[0], lm[1]] = sift_response[i]
        #     landmark_map[lm[0], lm[1]] = 1

        return {'raw_imgs': item['image'],
                'ID': item['ID'],
                'sift_landmark': sift_landmark,
                # 'sift_response': response_map,
                # 'sift_descript': descript_map
                }

    def __getitem__(self, index):
        if self.retfunc == 1:
            return self.retfunc1(index)
        elif self.retfunc == 3:
            raise ValueError


    def __len__(self):
        return len(self.list)



if __name__ == "__main__":
    dataset = Cephalometric()
    data = dataset.__getitem__(1)
    import ipdb; ipdb.set_trace()