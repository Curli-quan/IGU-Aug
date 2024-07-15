# coding: utf-8
"""
    To prove that the variance exactly increased by entropy map
"""


import os.path
import random
import numpy as np
from sc.ssl2.ssl_probmap7 import *
from utils.tester.tester_gtv import Tester
from tutils import print_dict
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='ssl_probmap')
parser.add_argument('--config', default="configs/ssl/ssl_pretrain.yaml")
# parser.add_argument('--func', default="test")
args = trans_args(parser)
logger, config = trans_init(args, file=__file__, ex_config=EX_CONFIG)


def get_res3(pth, learner):
    tester = Tester(logger, config, retfunc=3, split='Train', cj_brightness=0.8, cj_contrast=0.6)
    learner.load(pth) if pth is not None else None
    learner.cuda()
    learner.eval()
    res = tester.test_func4(learner)
    return res


def reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)


def figure1():
    reproducibility(0)
    # fig = plt.figure()
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    fig.set_figwidth(10)
    fig.set_figheight(4)

    switch = False
    pths = []
    pths += ['baseline_ps64_3']
    length = np.array([1,1,1,1,1,1])
    for pth in pths:
        epochs = [450]
        for epoch in epochs:
            data_path = f"./tmp/tester_gtv3/ssl_{pth}_gtv1_epoch_{epoch}.npy"
            res1 = np.load(data_path)
            print(res1)
            # import ipdb; ipdb.set_trace()
            ax.plot(np.arange(0,7), res1, label=f"{epoch} " + pth, color="red")

    # loss_value = [22.85, 23.02, 23.15, 23.23, 23.35]
    # loss_label = np.array([0,2,3,4,4.5]) + 1
    # ax2.plot(loss_label, loss_value, label='training loss', color="green")
    # plt.legend()


    # pths += ['baseline_ps64_3']
    for pth in pths:
        epochs = [450]
        for epoch in epochs:
            data_path = f"./tmp/tester_gtv4/ssl_{pth}_gtv_epoch_{epoch}.npy"
            res1 = np.load(data_path)
            print(res1)
            ax2.plot(np.arange(0,1), res1[0:1], label=f"PSI")
            ax2.plot(np.arange(0,7), res1, label=f"DDI", color="green")

    ax2.legend(loc='upper left')
    ax.set_xlabel("IIE")
    ax.set_ylabel("PSI")
    ax2.set_ylabel("DDI")
    plt.savefig("./tmp/paper_fig1.png")
    plt.savefig("./tmp/paper_fig1.pdf")


# def figure2():

figure1()