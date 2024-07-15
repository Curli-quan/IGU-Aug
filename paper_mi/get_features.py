import torch
# from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model
from tutils.new.trainer import DDPTrainer, Trainer, Monitor, LearnerWrapper, LearnerModule
from tutils.new.manager import trans_args, trans_init
from tutils import tfilename, print_dict
import argparse
from torch import optim
# from utils.tester.tester_ssl import Tester
from utils.tester.tester_ssl_debug import Tester
from datasets.ceph.ceph_ssl import Cephalometric
from models.network_emb_study import UNet_Pretrained
from sc.ssl.ssl import Learner
import numpy as np
from einops import rearrange

def test(logger, config):
    import glob
    import cv2
    # tester = Tester(logger, config, split="Test1+2")
    tester = Tester(logger, config, split="Train")

    learner = Learner(logger=logger, config=config)
    learner.load(tfilename("/home1/quanquan/code/landmark/code/runs/ssl/ssl/baseline_ps64/", 'ckpt_v', 'model_latest.pth'))
    learner.cuda()
    learner.eval()
    
    dirpath = "/home1/quanquan/code/landmark/code/tproj/paper_mi/auged_over_patches"
    paths = glob.glob(dirpath + "/*.png")
    name = dirpath.split("/")[-1]
    all_feas = []
    for p in paths:
        im = cv2.imread(p)
        x = torch.Tensor(im)[None,:,:,:]
        x = rearrange(x, "b h w c -> b c h w")
        x = x.cuda()

        features_tmp = learner.forward(x)
        landmark = [32,32]
        
        feature_list = list()
        for id_depth in range(6):
            tmpl_y, tmpl_x = landmark[1] // (2 ** (5 - id_depth)), landmark[0] // (2 ** (5 - id_depth))
            # print(id_depth, tmpl_y, tmpl_x, features_tmp[id_depth].shape)
            mark_feature = features_tmp[id_depth]
            # print("1")
            mark_feature = mark_feature[0, :, tmpl_y, tmpl_x]
            mark_feature = mark_feature / ((mark_feature**2).sum()**0.5 + 1e-8)
            feature_list.append(mark_feature.detach().squeeze().cpu().numpy())
        feature_list = np.array(feature_list)
        # import ipdb; ipdb.set_trace()
        if np.isnan(feature_list.sum()):
            continue
        all_feas.append(feature_list)
    print(len(all_feas))
    np.save(f"paper_mi/tmp_features/{name}.npy", all_feas)
    print(name)
    # import ipdb; ipdb.set_trace()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/ssl/ssl.yaml")
    parser.add_argument("--func", default="test")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    # save_script(config['base']['runs_dir'], __file__)
    print_dict(config)

    eval(args.func)(logger, config)
