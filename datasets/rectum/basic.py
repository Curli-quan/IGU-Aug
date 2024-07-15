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
# from datasets.augment import cc_augment
# from utils.entropy_loss import get_guassian_heatmaps_from_ref
import json
import glob
from trans_utils.excelsolver import ExcelSolver
from PIL import Image
from typing import Any
from einops import rearrange, repeat


class RectumDataset(data.Dataset):
    def __init__(self, 
                 dirpath="/home1/zhhli/sagjpg", 
                 excel_path="/home1/quanquan/code/landmark/code/tproj/trans_utils/landmark_gangmen.xlsx",
                 size=(384,384),
                 ) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.excel_path = excel_path
        self.size = size

        self.transform = transforms.Compose([transforms.Resize(self.size),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0], [1]),])
        self._prepare_dataset()

    def _prepare_dataset(self):
        self.solver = ExcelSolver()
        self.solver.set_file(self.excel_path)
        self.solver.read()

    def _get_data(self, index):
        label_dict = self.solver.get_label(index)
        filename = label_dict['name']
        img_name = os.path.join(self.dirpath, filename)
        img = Image.open(img_name).convert("RGB")
        assert img is not None
        return {"img":img, **label_dict}
    
    def resize_img_and_landmarks(self, d):
        ori_size = np.array(d['img']).shape[:2]
        new_dict = {}
        new_dict['name'] = d['name']
        new_dict['ori_size'] = ori_size
        img = self.transform(d['img'])
        assert img is not None
        
        new_dict['img'] = img
        # def resize_landmark(self, landmark):
        landmarks = []
        for k,v in d.items():
            if k.startswith("landmark_"):
                landmarks.append([v[1] * self.size[0] / ori_size[0], v[0] * self.size[1] / ori_size[1]])
        new_dict['landmarks'] = np.array(landmarks).astype(int)

        return new_dict
    
    def get_template(self, index):
        """
            Return [(h,w), ...]
        """
        if isinstance(index, list):
            data_collect = []
            for i in index:
                data = self._get_data(i)
                data = self.resize_img_and_landmarks(data)
                data_collect.append(data)
            
            # Remove these indices
            self.solver.remove_by_index(index)
            return data_collect
        else:
            data = self._get_data(index)
            data = self.resize_img_and_landmarks(data)
            self.solver.remove_by_index(index)
            return data

    # def visualize(self, d):
    #     Visualizer()
    def __len__(self):
        return len(self.solver)

    def __getitem__(self, index) -> Any:
        """
            Return [(h,w), ...]
        """
        d = self._get_data(index)
        d = self.resize_img_and_landmarks(d)
        return d
        
    def get_mask_by_name(self, name):
        img_name = os.path.join(self.dirpath.replace("sagjpg", "sagmask"), name)
        img = Image.open(img_name).convert("L")
        resize = transforms.Resize(self.size)
        img = resize(img)
        return np.array(img)


if __name__ == "__main__":
    from trans_utils.visualizer import Visualizer
    dataset = RectumDataset()
    mask = dataset.get_mask_by_name("10033831_06.jpg")
    # for i in range(len(dataset)):
    #     data = dataset.__getitem__(i)
    # data = dataset.__getitem__(0)

    import ipdb; ipdb.set_trace()

    vis = Visualizer()
    vis.display2(data)
