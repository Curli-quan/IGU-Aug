import torch
from tutils import tfilename, print_dict
from tutils.new.manager import  trans_args, trans_init, ConfigManager
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerModule
import argparse
from torch import optim
# from utils.tester.tester_ssl import Tester
from utils.tester.tester_ssl_debug import Tester
from datasets.ceph.ceph_ssl_adap import Cephalometric as Ceph_abstract
from models.network_emb_study import UNet_Pretrained
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sc.ssl2.ssl_aug4 import Learner

torch.backends.cudnn.benchmark = True


def test(logger, config):
    # tester = Tester(logger, config, split="Test1+2")
    tester = Tester(logger, config, split="Train")

    learner = Learner(logger=logger, config=config)
    learner.load("/home1/quanquan/code/landmark/code/runs/ssl/ssl/baseline_ps64/ckpt_v/model_latest.pth")
    # learner.load("/home1/quanquan/code/landmark/code/runs/ssl/ssl_aug4/6_class/ckpt_v/model_best.pth")
    learner.cuda()
    learner.eval()

    tester.draw_one_cosmap(learner, oneshot_id=114)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/ssl/ssl_pretrain.yaml")
    args = trans_args(parser)
    logger = None
    config = ConfigManager()
    config.add_basic_config()
    config.add_config(args)
    print_dict(config)
    test(logger, config)
