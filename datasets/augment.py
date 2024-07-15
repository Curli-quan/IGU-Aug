import numpy as np
import torch
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug

def cc_augment(data, \
    do_elastic_deform=True, alpha=(1000., 2000.), sigma=(10., 15.), do_rotation=True, angle_x=(-np.pi/9, np.pi/9), \
        angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi), do_scale=True, scale=(0.75, 1.25), \
            border_mode_data='constant', border_cval_data=0, order_data=3, border_mode_seg='constant',\
                 border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1, p_scale_per_sample=1,\
                      p_rot_per_sample=1, tag=''):
    
    data = data.numpy()
    shape = data.shape[1:]

    coords = create_zero_centered_coordinate_mesh(shape)

    # if np.random.uniform() < p_el_per_sample and do_elastic_deform:
    #     a = np.random.uniform(alpha[0], alpha[1])
    #     s = np.random.uniform(sigma[0], sigma[1])
    #     coords = elastic_deform_coordinates(coords, a, s)

    if np.random.uniform() < p_rot_per_sample and do_rotation:
        if angle_x[0] == angle_x[1]:
            a_x = angle_x[0]
        else:
            a_x = np.random.uniform(angle_x[0], angle_x[1])
        coords = rotate_coords_2d(coords, a_x)
    
    for i in range(len(shape)):
        coords[i] += shape[i] / 2
    for id in range(data.shape[0]):
        data[id] = interpolate_img(data[id], coords, 3, \
            border_mode_data, cval=border_cval_data)
    # data_result = (data_result > 0.1).astype(np.uint8)
    return torch.from_numpy(np.clip(data, 0, 1))


from torchvision import transforms
def crop_and_resize(tensor=None,size=(192, 192)):
    transform = transforms.RandomResizedCrop(size)
    a = torch.ones((3,384,384))
    a = transform(a)
    print(a.shape)

def interp():
    import torch.nn.functional as F
    aa = torch.ones((3,60,60))
    # a2 = F.interpolate(aa, size=(192,192), mode='linear', align_corners=True)
    transform = transforms.Resize((192,192))
    a2 = transform(aa)
    print(a2.shape)
    import ipdb;ipdb.set_trace()

if __name__ == '__main__':
    interp()