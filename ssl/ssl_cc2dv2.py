import torch
from tutils import trans_args, trans_init, save_script, tfilename, print_dict, count_model
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerModule
import argparse
from torch import optim
from utils.tester.tester_ssl_debug import Tester
from datasets.ceph.ceph_ssl import Cephalometric
from models.network_emb_study import UNet_Pretrained


torch.backends.cudnn.benchmark = True

cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-3)
def match_inner_product(feature, template):
    cos_sim = cosfn(template.unsqueeze(-1).unsqueeze(-1), feature)
    cos_sim = torch.clamp(cos_sim, 0., 1.)
    return cos_sim


def ce_loss(cos_map, gt_x, gt_y, nearby=None, add_bias=False):
    b, w, h = cos_map.shape
    total_loss = list()

    strength = 0.1
    temperature = 10
    clip = 0.7
    for id in range(b):
        # cos_map[id] = cos_map[id].exp()
        # gt_value = cos_map[id, gt_x[id], gt_y[id]].clone()
        if nearby is not None:
            min_y, max_y = max(gt_y[id] - nearby, 0), min(gt_y[id] + nearby, h)
            min_x, max_x = max(gt_x[id] - nearby, 0), min(gt_x[id] + nearby, w)
            chosen_patch = cos_map[id, min_y:max_y, min_x:max_x]
            gap_y, gap_x = gt_y[id] - min_y, gt_x[id] - min_x
            y, x = chosen_patch.shape
            if add_bias:
                mesh = torch.meshgrid(torch.linspace(0, y - 1, y), torch.linspace(0, x - 1, x))
                distance_y, distance_x = mesh[0].cuda() - gap_y, mesh[1].cuda() - gap_x
                distance = torch.sqrt(distance_y ** 2 + distance_x ** 2).cuda()
                distance = torch.clamp(distance * strength * temperature, 0, clip * temperature)
                chosen_patch += distance
            chosen_patch = chosen_patch.exp()
            gt_value = chosen_patch[gap_y, gap_x].clone()
        else:
            if add_bias:
                y, x = cos_map[id].shape
                mesh = torch.meshgrid(torch.linspace(0, y - 1, y), torch.linspace(0, x - 1, x))
                distance_y, distance_x = mesh[0].cuda() - gt_y[id], mesh[1].cuda() - gt_x[id]
                distance = torch.sqrt(distance_y ** 2 + distance_x ** 2).cuda()
                distance = torch.clamp(distance * strength * temperature, 0, clip * temperature)
                cos_map[id] += distance
            cos_map[id] = cos_map[id].exp()
            gt_value = cos_map[id, gt_y[id], gt_x[id]].clone()
            chosen_patch = cos_map[id]

        id_loss = - torch.log(gt_value / chosen_patch.sum())
        total_loss.append(id_loss)
    return torch.stack(total_loss).mean()


def _tmp_loss_func(ii, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=None, add_bias=False):
    scale = 2 ** (5-ii)
    raw_loc, chosen_loc = raw_loc // scale, chosen_loc // scale
    tmpl_feature = torch.stack([crop_fea_list[ii][[id], :, chosen_loc[id][0], chosen_loc[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    product = match_inner_product(raw_fea_list[ii], tmpl_feature)  # shape [8,12,12]
    loss = ce_loss(product, raw_loc[:, 0], raw_loc[:, 1], nearby=nearby, add_bias=add_bias)
    return loss


class Learner(LearnerModule):
    def __init__(self, logger, config, *args, **kwargs):
        super(Learner, self).__init__(*args, **kwargs)
        self.logger = logger
        self.config = config
        self.net = UNet_Pretrained(3, non_local=config['special']['non_local'], emb_len=16)
        self.net_patch = UNet_Pretrained(3, emb_len=16)
        self.loss_logic_fn = torch.nn.CrossEntropyLoss()
        self.mse_fn = torch.nn.MSELoss()


    def forward(self, x, **kwargs):
        # self.net(x['img'])
        # raise NotImplementedError
        return self.net(x)

    def training_step(self, data, batch_idx, **kwargs):
        raw_imgs = data['raw_imgs']
        crop_imgs = data['crop_imgs']
        raw_loc  = data['raw_loc']
        chosen_loc = data['chosen_loc']

        raw_fea_list = self.net(raw_imgs)
        crop_fea_list = self.net_patch(crop_imgs)

        nearby = self.config['special']['nearby']
        loss_0 = _tmp_loss_func(0, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, add_bias=True)
        loss_1 = _tmp_loss_func(1, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby, add_bias=True)
        loss_2 = _tmp_loss_func(2, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby, add_bias=True)
        loss_3 = _tmp_loss_func(3, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby, add_bias=True)
        loss_4 = _tmp_loss_func(4, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby, add_bias=True)

        loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4

        return {'loss': loss, 'loss_0': loss_0, 'loss_1': loss_1, 'loss_2': loss_2, 'loss_3': loss_3, 'loss_4': loss_4}

    def load(self, pth=None, *args, **kwargs):
        if pth is None:
            print("Load Pretrain Model")
            state_dict = torch.load(self.config['network']['pretrain'])
            self.net.load_state_dict(state_dict)
        else:
            print("Load Pretrain Model:", pth)
            state_dict = torch.load(pth)
            self.net.load_state_dict(state_dict)


    def save_optim(self, pth, optimizer, epoch, *args, **kwargs):
        pass

    def configure_optimizers(self, *args, **kwargs):
        config_train = self.config['training']
        optimizer = optim.Adam(params=self.net.parameters(), lr=config_train['lr'], betas=(0.9, 0.999),
                               eps=1e-8, weight_decay=config_train['weight_decay'])
        optimizer_patch = optim.Adam(params=self.net_patch.parameters(), lr=config_train['lr'],
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=config_train['weight_decay'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, config_train['decay_step'], gamma=config_train['decay_gamma'])
        scheduler_patch = optim.lr_scheduler.StepLR(optimizer_patch, config_train['decay_step'], gamma=config_train['decay_gamma'])
        return {'optimizer': [optimizer, optimizer_patch], 'scheduler': [scheduler, scheduler_patch]}


def train(logger, config):
    from datasets.ceph.ceph_ssl import Test_Cephalometric
    tester = Tester(logger, config, collect_sim=True)
    monitor = Monitor(key='mre', mode='dec')

    id_oneshot = 114
    testset = Test_Cephalometric(config['dataset']['pth'], mode="Train", pre_crop=False)
    landmark_list = testset.ref_landmarks(id_oneshot)
    dataset_train = Cephalometric(config['dataset']['pth'], patch_size=config['special']['patch_size'],
                                  pre_crop=False, ref_landmark=landmark_list, use_prob=True, retfunc=2)
    trainer = Trainer(logger=logger, config=config, tester=tester, monitor=monitor)
    learner = Learner(logger=logger, config=config)
    trainer.fit(learner, dataset_train)


def test(logger, config):
    # tester = Tester(logger, config, split="Test1+2")
    tester = Tester(logger, config, split="Train")

    learner = Learner(logger=logger, config=config)
    learner.load(tfilename(config['base']['runs_dir'], 'ckpt', 'best_model_epoch_400.pth'))
    learner.cuda()
    learner.eval()

    # ids = [1,2,3,4,5,6,7,8,9]
    ids = [114, 124, 125, ]
    for id_oneshot in ids:
        res = tester.test(learner, oneshot_id=id_oneshot)
        logger.info(res)
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='ssl_probmap')
    parser.add_argument('--config', default="configs/ssl/v2.yaml")
    # parser.add_argument('--func', default="test")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)
    # save_script(config['base']['runs_dir'], __file__)
    # print_dict(config)
    print(config['base'])

    eval(args.func)(logger, config)
