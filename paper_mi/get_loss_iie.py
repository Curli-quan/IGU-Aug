import torch
# from tutils import save_script, tfilename, print_dict, count_model, CSVLogger
from tutils.trainer import DDPTrainer, Trainer, Monitor, LearnerWrapper
import argparse
from torch import optim
from utils.tester.tester_hand_ssl import Tester
# from datasets.hand.hand_ssl import HandXray
from datasets.hand.hand_ssl_adapm import HandXray
from models.network_emb_study import UNet_Pretrained
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd


EX_CONFIG = {
    "special": {
        # "pretrain_model": "/home1/quanquan/code/landmark/code/runs/ssl/ssl_hand/run1/ckpt_v/model_latest.pth",
        "patch_size": 192,
        "prob_ratio": 0,
        "entr_t": 0.5,
        "cj_brightness": [0.7,1.5],
        "cj_contrast": [0.7,1.3],
    },
    "training": {
        "load_pretrain_model": False,
        'val_check_interval': 10,
        'num_epochs' : 450,  # epochs
    }
}


torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')

cosfn = torch.nn.CosineSimilarity(dim=1, eps=1e-3)
def match_inner_product(feature, template):
    cos_sim = cosfn(template.unsqueeze(-1).unsqueeze(-1), feature)
    cos_sim = torch.clamp(cos_sim, 0., 1.)
    return cos_sim

def ce_loss(cos_map, gt_x, gt_y, nearby=None):
    b, w, h = cos_map.shape
    total_loss = list()
    gt_values_to_record = []
    for id in range(b):
        cos_map[id] = cos_map[id].exp()
        gt_value = cos_map[id, gt_x[id], gt_y[id]].clone()
        if nearby is not None:
            min_x, max_x = max(gt_x[id] - nearby, 0), min(gt_x[id] + nearby, w)
            min_y, max_y = max(gt_y[id] - nearby, 0), min(gt_y[id] + nearby, h)
            chosen_patch = cos_map[id, min_x:max_x, min_y:max_y]
        else:
            chosen_patch = cos_map[id]
        id_loss = - torch.log(gt_value / chosen_patch.sum())
        gt_values_to_record.append(gt_value.clone().detach().log().cpu())
        total_loss.append(id_loss)
    gt_values_to_record = torch.stack(gt_values_to_record).mean()
    return torch.stack(total_loss).mean(), gt_values_to_record


def _tmp_loss_func(ii, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby=None):
    scale = 2 ** (5-ii)
    raw_loc, chosen_loc = raw_loc // scale, chosen_loc // scale
    tmpl_feature = torch.stack([crop_fea_list[ii][[id], :, chosen_loc[id][0], chosen_loc[id][1]] \
                                for id in range(raw_loc.shape[0])]).squeeze()
    product = match_inner_product(raw_fea_list[ii], tmpl_feature)  # shape [8,12,12]
    loss, gt_values = ce_loss(product, raw_loc[:, 0], raw_loc[:, 1], nearby=nearby)
    return loss, gt_values


class Learner(LearnerWrapper):
    def __init__(self, logger, config, *args, **kwargs):
        super(LearnerWrapper, self).__init__(*args, **kwargs)
        self.logger = logger
        self.config = config
        self.net = UNet_Pretrained(3, non_local=config['special']['non_local'], emb_len=config['special']['emb_len'])
        self.net_patch = UNet_Pretrained(3, emb_len=config['special']['emb_len'])
        self.loss_logic_fn = torch.nn.CrossEntropyLoss()
        self.mse_fn = torch.nn.MSELoss()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        self.net.train()
        self.net_patch.train()

    def eval(self):
        self.net.eval()
        self.net_patch.eval()

    def cuda(self):
        self.net.cuda()
        self.net_patch.cuda()

    def to(self, rank):
        self.net.to(rank)
        self.net_patch.to(rank)

    def forward(self, x, **kwargs):
        # self.net(x['img'])
        # raise NotImplementedError
        return self.net(x)

    def training_step(self, data, batch_idx, **kwargs):
        raw_imgs = data['raw_imgs']
        crop_imgs = data['crop_imgs']
        raw_loc  = data['raw_loc']
        chosen_loc = data['chosen_loc']
        entr = data['entr']

        raw_fea_list = self.net(raw_imgs)
        crop_fea_list = self.net_patch(crop_imgs)

        nearby = self.config['special']['nearby']
        loss_0, gt_value_0 = _tmp_loss_func(0, raw_fea_list, crop_fea_list, raw_loc, chosen_loc)
        loss_1, gt_value_1 = _tmp_loss_func(1, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_2, gt_value_2 = _tmp_loss_func(2, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_3, gt_value_3 = _tmp_loss_func(3, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)
        loss_4, gt_value_4 = _tmp_loss_func(4, raw_fea_list, crop_fea_list, raw_loc, chosen_loc, nearby)

        loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4
        loss_dict = {'loss': loss, 'loss_0': loss_0, 'loss_1': loss_1, 'loss_2': loss_2, 'loss_3': loss_3,
                     'loss_4': loss_4}
        gtvs_dict = {'gtv_0': gt_value_0, 'gtv_1': gt_value_1, 'gtv_2': gt_value_2, 'gtv_3': gt_value_3,
                     'gtv_4': gt_value_4}
        # res_dict = {**loss_dict, **gtvs_dict}
        res_dict = {"loss": loss, "entr": entr}

        return res_dict

    def load(self, path=None, **kwargs):
        if path is None:
            path = self.config['special']['pretrain_model']
        state_dict = torch.load(path)
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


class MyTrainer(Trainer):
    def init_model(self, model, trainset, validset=None, **kwargs):
        if trainset is not None:
            assert len(trainset) > 0 , f"Got {len(trainset)}"
            self.trainloader = DataLoader(dataset=trainset,
                                        batch_size=1,
                                        num_workers=self.num_workers,
                                        shuffle=False,
                                        drop_last=True,
                                        pin_memory=True)
        if self.load_pretrain_model:
            model.load()
        model.net = torch.nn.DataParallel(model.net)
        model.cuda()
        return model
    
    def get_info(self, model, trainset, validset=None):
        model = self.init_model(model, trainset, validset=validset, rank=self.rank)
        self.init_timers()
        optimizer, scheduler, start_epoch = self.configure_optim(model)

        for epoch in range(start_epoch, 50):
            self.train(model, self.trainloader, epoch, optimizer, scheduler)
            print(f"epoch: {epoch}", end="\r")
        
        df = pd.DataFrame({
            "loss": [d[0] for d in self.recorder.loss_list],
            "entr": [d[1] for d in self.recorder.loss_list],
        })
        # self.recorder.loss_list
        df.to_csv("loss_iie_hand.csv")
        import ipdb; ipdb.set_trace()
    
    def train(self, model, trainloader, epoch, optimizer, scheduler=None, do_training_log=True):
        model.eval()
        out = {}
        
        for batch_idx, data in enumerate(trainloader):
            model.on_before_zero_grad()
            optimizer.zero_grad()
            self.timer_data()
            # training steps
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(self.rank)
            time_data_cuda = self.timer_data()
            out = model.training_step(data, batch_idx, epoch=epoch)
            assert isinstance(out, dict)
            time_fd = self.timer_net()

            self.recorder.record(out)   
            if batch_idx > 16:
                break


def train(logger, config):
    tester = Tester(logger, config, split="Test", upsample="nearest")
    monitor = Monitor(key='mre', mode='dec')
    dataset_train = HandXray(
            config['dataset']['pth'],
            patch_size=config['special']['patch_size'],
            mode="Train",
            num_repeat=1)
    dataset_train.entr_map_from_image()
    trainer = MyTrainer(logger=logger, config=config, tester=tester, monitor=monitor)
    learner = Learner(logger=logger, config=config)
    learner.load("/home1/quanquan/code/landmark/code/runs/ssl/ssl_hand/baseline/ckpt_v/model_best.pth")
    learner.cuda()
    learner.eval()
    trainer.get_info(learner, dataset_train)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    # from tutils import TConfig
    from tutils.new.manager import ConfigManager, trans_args, trans_init
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/ssl/ssl_hand.yaml")
    parser.add_argument('--idnum', type=int, default=1)
    
    args = trans_args(parser=parser)
    # config = ConfigManager()
    # config.add_basic_config()
    # config.add_config(args)
    # config.init()
    # logger = None
    logger, config = trans_init(args, file=__file__)
    train(logger, config)
