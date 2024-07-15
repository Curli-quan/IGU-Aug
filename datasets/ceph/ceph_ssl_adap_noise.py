from .ceph_ssl_adap import Cephalometric as basic_dataset
from tutils import tfilename
import os
from PIL import Image
import cv2
import numpy as np


def add_salt_noise(image):
    #设置添加椒盐噪声的数目比例
    s_vs_p = 0.5
    #设置添加噪声图像像素的数目
    amount = 0.04
    noisy_img = np.copy(image)
    #添加salt噪声
    num_salt = np.ceil(amount * image.size * s_vs_p)
    #设置添加噪声的坐标位置
    coords = [np.random.randint(0,i - 1, int(num_salt)) for i in image.shape]
    noisy_img[coords[0],coords[1],:] = [255,255,255]
    #添加pepper噪声
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    #设置添加噪声的坐标位置
    coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in image.shape]
    noisy_img[coords[0],coords[1],:] = [0,0,0]
    #保存图片
    # cv2.imwrite("noisy_img.png",noise_img)
    return noisy_img

def add_speckle_noise(image):
    img_height,img_width,img_channels = image.shape
    #随机生成一个服从分布的噪声
    gauss = np.random.randn(img_height,img_width,img_channels)
    #给图片添加speckle噪声
    noisy_img = image + image * gauss
    #归一化图像的像素值
    noisy_img = np.clip(noisy_img,a_min=0,a_max=255)
    #保存图片
    # cv2.imwrite("noisy_img.png",noise_img)
    return noisy_img


class Cephalometric(basic_dataset):
    def __init__(self, pathDataset, datatag=None, split='noise', *args, **kwargs):
        super().__init__(pathDataset=pathDataset, split="Train", *args, **kwargs)
        self.pth_Image = os.path.join(pathDataset, 'RawImage')
        self.pth_Image = os.path.join(self.pth_Image, datatag)
        self.datatag = datatag
        print(self.pth_Image)

    def generate_noisy_images(self):
        self.save_path = self.pth_Image
        self.pth_Image = self.pth_Image.replace(self.datatag, 'TrainingData')

        for index in range(150): 
            item = self.list[index]
            name = item['ID'] + '.bmp'
            pth_img = os.path.join(self.pth_Image, name)
            image = Image.open(pth_img).convert('RGB')

            image = np.array(image)
            ori_shape = image.shape

            if index < 113:
                image = cv2.resize(image, (384,384))
                image = cv2.GaussianBlur(image, (11,11), 0)
                noisy_image = add_salt_noise(image)
                noisy_image = cv2.resize(noisy_image, (ori_shape[1], ori_shape[0]))
                cv2.imwrite(tfilename(self.save_path+"/"+name), noisy_image)
            else:
                cv2.imwrite(tfilename(self.save_path+"/"+name), image)
            print(self.save_path+"/"+name)



if __name__ == "__main__":
    dirpath = '/home1/quanquan/datasets/Cephalometric/'
    dataset = Cephalometric(pathDataset=dirpath, datatag="NoisyTrainingData75")
    dataset.generate_noisy_images()


        