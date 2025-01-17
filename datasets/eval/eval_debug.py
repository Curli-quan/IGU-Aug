import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
from pathlib import Path

def make_dir(pth):
    dir_pth = Path(pth)
    if not dir_pth.exists():
        dir_pth.mkdir()
    return pth


class Evaluater(object):
    def __init__(self, logger, size, original_size, tag='paper_figure', collect_details=False):
        self.pixel_spaceing = 0.1
        self.tag = tag
        make_dir(tag)
        self.tag += '/'

        # self.logger = logger
        self.logger = None
        self.scale_rate_y = original_size[0] / size[0]
        self.scale_rate_x = original_size[1] / size[1]

        self.RE_list = list()

        self.recall_radius = [2, 2.5, 3, 4]  # 2mm etc
        self.recall_rate = list()

        self.Attack_RE_list = list()
        self.Defend_RE_list = list()

        self.dict_Attack = dict()
        self.dict_Defend = dict()
        self.total_list = dict()

        self.mode_list = [0, 1, 2, 3]
        self.mode_dict = {0: "Iterative FGSM", 1: "Adaptive Iterative FGSM", \
                          2: "Adaptive_Rate", 3: "Proposed"}

        for mode in self.mode_list:
            self.dict_Defend[mode] = dict()
            self.dict_Attack[mode] = dict()
            self.total_list[mode] = list()
        self.best_mre = 100.0

    def info(self, msg, *args, **kwargs):
        pass
        # if self.logger is not None:
        #     self.logger.info(msg)
        # else:
        #     print(msg, *args, **kwargs)

    def reset(self):
        self.RE_list.clear()
        for mode in self.mode_list:
            self.dict_Defend[mode] = dict()
            self.dict_Attack[mode] = dict()
            self.total_list[mode] = list()
        self.Attack_RE_list.clear()
        self.Defend_RE_list.clear()

    def record_old(self, pred, landmark):
        """
            pred:　[h,h,...], [w,w,...]
            landmark: [(w,h), (w,h), ...]
        """
        c = pred[0].shape[0]
        assert c == 19, f"Shape error!, Got {pred.shape}"
        diff = np.zeros([c, 2], dtype=float)  # y, x
        for i in range(c):
            diff[i][0] = abs(pred[0][i] - landmark[i][1]) * self.scale_rate_y
            diff[i][1] = abs(pred[1][i] - landmark[i][0]) * self.scale_rate_x
        Radial_Error = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
        Radial_Error *= self.pixel_spaceing
        self.RE_list.append(Radial_Error)
        return Radial_Error

    def record(self, *args, **kwargs):
        raise TypeError("This function is deprecated! Use [eval.record_new(), or record_old()] instead!")

    def record_new(self, pred, landmark):
        """
            pred:　[(h,w), (h,w), ...]
            landmark: [(h,w), (h,w), ...]
        """
        pred = np.array(pred)
        landmark = np.array(landmark)
        assert pred.shape[1] == 2 and landmark.shape[1] == 2
        c = pred.shape[0]
        diff = np.zeros_like(pred, dtype=float)  # y, x
        for i in range(c):
            diff[i][0] = abs(pred[i][0] - landmark[i][0]) * self.scale_rate_y
            diff[i][1] = abs(pred[i][1] - landmark[i][1]) * self.scale_rate_x
        Radial_Error = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
        Radial_Error *= self.pixel_spaceing
        self.RE_list.append(Radial_Error)
        return Radial_Error


    def cal_metrics(self, return_sdr=False):
        # calculate MRE SDR
        temp = np.array(self.RE_list)
        Mean_RE_channel = temp.mean(axis=0)
        # self.info(Mean_RE_channel)
        mre = Mean_RE_channel.mean()
        # self.info("ALL MRE {}".format(mre))

        sdr_dict = {}
        for radius in self.recall_radius:
            total = temp.size
            shot = (temp < radius).sum()
            sdr_dict[f"SDR {radius}"] = shot * 100 / total
            # self.info("ALL SDR {}mm  {}".format\
            #                      (radius, shot * 100 / total))
        if return_sdr:
            return {'mre':mre, **sdr_dict}
        return {'mre':mre}

    def cal_metrics_all(self):
        return self.cal_metrics(return_sdr=True)

    def cal_metrics_details(self):
        temp = np.array(self.RE_list)
        Mean_RE_channel = temp.mean(axis=0)
        mre = Mean_RE_channel.mean()

        sdr_dict = {}
        for radius in self.recall_radius:
            total = temp.size
            shot = (temp < radius).sum()
            sdr_dict[f"SDR {radius}"] = shot * 100 / total
            # self.info("ALL SDR {}mm  {}".format\
            #                      (radius, shot * 100 / total))
        return {'mre': mre, **sdr_dict, "diff": temp}