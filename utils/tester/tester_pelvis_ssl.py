import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from PIL import Image, ImageDraw, ImageFont
from datasets.pelvis.pelvis_basic import Pelvis
from datasets.eval.eval_pelvis import Evaluater
from utils.utils import visualize

from tutils import tfilename, tdir


def gray_to_PIL(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images


def gray_to_PIL2(tensor, pred_lm, landmark, row=6, width=384):
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(images)
    red = (255, 0, 0)
    green = (0, 255, 0)
    # red = 255
    for i in range(row):
        draw.rectangle((pred_lm[0] + i * width - 2, pred_lm[1] - 2, pred_lm[0] + i * width + 2, pred_lm[1] + 2),
                       fill=green)
        draw.rectangle((landmark[0] + i * width - 2, landmark[1] - 2, landmark[0] + i * width + 2, landmark[1] + 2),
                       fill=red)
    draw.line([tuple(pred_lm), tuple(landmark)], fill='green', width=0)
    # import ipdb; ipdb.set_trace()
    return images


def match_cos(feature, template):
    feature = feature.permute(1, 2, 0)
    fea_L2 = torch.norm(feature, dim=-1)
    template_L2 = torch.norm(template, dim=-1)
    inner_product = (feature * template).sum(-1)
    cos_similarity = inner_product / (fea_L2 * template_L2 + 1e-3)
    return torch.clamp(cos_similarity, 0, 1)


class Tester(object):
    def __init__(self, logger, config, mode="Test", args=None):
        dataset_1 = Pelvis(config['dataset']['pth'], mode=mode)
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                       shuffle=False, num_workers=config['training']['num_workers'])
        self.one_shot_loader = Pelvis(pathDataset=config['dataset']['pth'], \
                                 mode='Train')
        self.logger = logger
        self.config = config
        self.args = args
        # Creat evluater to record results
        self.evaluater = Evaluater(None, [384, 384],
                                   [2544, 3056])
        self.dataset = dataset_1
        self.id_landmarks = [i for i in range(config['dataset']['n_cls'])]

    def test(self, *args, **kwargs):
        return self.test_func(*args, **kwargs)

    def test_func(self, net, epoch=None, rank='cuda', oneshot_id=25, dump_label=False, draw=False):
        net.eval()
        config = self.config
        print(f'ID Oneshot : {oneshot_id}')
        self.evaluater.reset()
        data = self.one_shot_loader.__getitem__(oneshot_id)
        image, landmarks, index = data['img'], data['landmark_list'], data['index']
        feature_list = list()
        if rank != 'cuda':
            image = image.to(rank)
        else:
            image = image.cuda()
        features_tmp = net(image.unsqueeze(0))

        num_landmarks = 10
        # Depth
        feature_list = dict()
        for id_depth in range(6):
            tmp = list()
            for id_mark, landmark in enumerate(landmarks):
                tmpl_y, tmpl_x = landmark[1] // (2 ** (5 - id_depth)), landmark[0] // (2 ** (5 - id_depth))
                mark_feature = features_tmp[id_depth][0, :, tmpl_y, tmpl_x]
                tmp.append(mark_feature.detach().squeeze().cpu().numpy())
            tmp = np.stack(tmp)
            if rank != 'cuda':
                one_shot_feature = torch.tensor(tmp).to(rank)
            else:
                one_shot_feature = torch.tensor(tmp).cuda()
            feature_list[id_depth] = one_shot_feature

        ID = 1

        for data in self.dataloader_1:
            if rank != 'cuda':
                img = data['img'].to(rank)
            else:
                img = data['img'].cuda()
            features = net(img)
            landmark_list = data['landmark_list'][0]

            pred_landmarks_y, pred_landmarks_x = list(), list()
            for id_mark in range(num_landmarks):
                cos_lists = []
                cos_ori_lists = []
                if rank != 'cuda':
                    final_cos = torch.ones_like(img[0, 0]).to(rank)
                else:
                    final_cos = torch.ones_like(img[0, 0]).cuda()
                for id_depth in range(5):
                    cos_similarity = match_cos(features[id_depth].squeeze(), \
                                               feature_list[id_depth][id_mark])
                    # import ipdb;ipdb.set_trace()
                    cos_similarity = torch.nn.functional.upsample( \
                        cos_similarity.unsqueeze(0).unsqueeze(0), \
                        scale_factor=2 ** (5 - id_depth), mode='nearest').squeeze()
                    # import ipdb;ipdb.set_trace()
                    final_cos = final_cos * cos_similarity
                    cos_lists.append(cos_similarity)
                    # import ipdb;ipdb.set_trace()
                final_cos = (final_cos - final_cos.min()) / \
                            (final_cos.max() - final_cos.min())
                cos_lists.append(final_cos)
                ## TODO: Here should be changed to unravel_index
                chosen_landmark = final_cos.argmax().item()
                pred_landmarks_y.append(chosen_landmark // 384)
                pred_landmarks_x.append(chosen_landmark % 384)
                ##
                debug = torch.cat(cos_lists, 1).cpu()

                a_landmark = landmark_list[id_mark]
                pred_landmark = (chosen_landmark % 384, chosen_landmark // 384)
                if draw:
                    gray_to_PIL2(debug.cpu().detach(), pred_landmark, a_landmark).save( \
                        tfilename(config['base']['runs_dir'], 'visuals', str(ID), f'{id_mark + 1}_debug_w_gt.jpg'))

            preds = [np.array(pred_landmarks_y), np.array(pred_landmarks_x)]
            self.evaluater.record(preds, landmark_list)
            # image_pred = visualize(img, pred_landmark, landmark_list)
            # image_pred.save(os.path.join('visuals', str(ID) + '_pred.png'))

            # Optional Save viusal results
            if draw:
                image_pred = visualize(img, preds, landmark_list, num=10)
                image_pred.save(tfilename(config['base']['runs_dir'], 'visuals', str(ID), 'pred.png'))

            if dump_label:
                inference_marks = {id: [int(preds[1][id]), \
                                        int(preds[0][id])] for id in range(19)}
                dir_pth = tdir(config['base']['runs_dir'], 'pseudo_labels_init')
                with open('{0}/{1:03d}.json'.format(dir_pth, ID), 'w') as f:
                    json.dump(inference_marks, f)
                print("Dumped JSON file:", '{0}/{1:03d}.json'.format(dir_pth, ID), end="\r")

            ID += 1

        mre = self.evaluater.cal_metrics_all()
        return mre
