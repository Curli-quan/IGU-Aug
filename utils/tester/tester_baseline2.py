"""
    Basic Tester with regression module

"""
from datasets.eval.eval import Evaluater
from datasets.ceph.ceph_test import Test_Cephalometric
from utils.utils_st import voting
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# from torchvision.utils import save_image
from einops import rearrange
from tutils import  tfilename
from utils.utils import visualize


class Tester(object):
    def __init__(self, logger, config, args=None, split='Test1+2', get_mre_per_lm=False):

        dataset_1 = Test_Cephalometric(config['dataset']['pth'], mode=split)
        self.split = split
        self.dataloader = DataLoader(dataset_1, batch_size=1,
                                       shuffle=False, num_workers=2)
        self.Radius = dataset_1.Radius
        self.config = config
        self.evaluater = Evaluater(logger, [384, 384],
                                       [2400, 1935])
        self.logger = logger

        self.dataset = dataset_1
        self.id_landmarks = [i for i in range(config['dataset']['n_cls'])]
        self.get_mre_per_lm = get_mre_per_lm

    def test(self, model, epoch=1, rank=-1, draw=False):
        self.evaluater.reset()
        model.eval()
        ID = 1
        runs_dir = self.config['base']['runs_dir']
        for data in tqdm(self.dataloader, ncols=70):
            if rank != 'cuda' and rank >= 0:
                img = data['img'].to(rank)
            else:
                img = data['img'].cuda()
            landmark_list = data['landmark_list']

            heatmap, regression_y, regression_x = model(img)

            # gray_to_PIL(heatmap[0][1].cpu().detach()) \
            #     .save(os.path.join('visuals', str(ID) + '_heatmap.png'))
            # Vote for the final accurate point

            pred_landmark, votings = voting( \
                heatmap, regression_y, regression_x, self.Radius, get_voting=True)

            self.evaluater.record_old(pred_landmark, landmark_list)

            if draw:
                # Optional Save viusal results
                print(f"Draw img: {tfilename(runs_dir, 'visuals', str(ID) + '_pred.png')}")
                image_pred = visualize(img, pred_landmark, landmark_list, num=19)
                image_pred.save(tfilename(runs_dir, 'visuals', str(ID) + '_pred.png'))

            ID += 1
            if epoch == 0:
                print("for DEBUG")
                break

        if self.get_mre_per_lm:
            _d = self.evaluater.cal_metrics_per_lm()
            _d['split'] = self.split
            return _d
        # return {"mre": mre, "sdr": sdr, ...}
        _d = self.evaluater.cal_metrics_per_lm()
        _d['split'] = self.split
        # _d['testset'] = self.testset
        return _d
    
    def draw(self, model, *args, **kwargs):
        return self.test(model, draw=True)

    def debug(self, model):
        print("DEBUG")
        model.eval()
        self.evaluater.reset()
        for data in self.dataloader:
            print(data['name'])
            img = data['img'].cuda()
            landmark_list = data['landmark_list']
            heatmap, regression_y, regression_x = model(img, return_features=True)
            break
        print("DEBUG")


    def dump_pseudo_dataset(self, model, iteration=1):
        model.eval()
        ID = 1

        dataset = Test_Cephalometric(self.config['dataset']['pth'], mode='Train')
        trainloader = DataLoader(dataset, batch_size=1,
                                       shuffle=False, num_workers=2)

        for i, data in tqdm(enumerate(trainloader), ncols=100):
            img = data['img'].cuda()
            landmark_list = data['landmark_list']

            heatmap, regression_y, regression_x = model(img)
            pred_landmark, votings = voting( \
                heatmap, regression_y, regression_x, self.Radius, get_voting=True)
            self.evaluater.record_old(pred_landmark, landmark_list)
            pred_landmark = np.array(pred_landmark).transpose((1, 0))
            # import ipdb; ipdb.set_trace()
            np.save(tfilename(self.config['base']['runs_dir'], "pseudo_labels", f"iter_{iteration}", f"{ID}.npy"), np.array(pred_landmark))
            if i <= 0:
                print(f" shape {np.array(pred_landmark).shape}")
                print("[] Np.save ", f"iter_{iteration}/" + f"{ID}.npy")
            ID += 1
        return self.evaluater.cal_metrics_all()
