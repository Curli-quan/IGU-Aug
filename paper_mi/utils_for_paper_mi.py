import numpy as np
from skimage.exposure import histogram


from sklearn.metrics import mutual_info_score

def get_fea(patch):
    fea = np.zeros((256,))
    hist, idx = histogram(patch, nbins=256)
    for hi, idi in zip(hist, idx):
        # print(hi, idi, i, j)
        fea[idi] = hi
    return fea