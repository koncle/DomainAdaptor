import copy
from pathlib import Path

import numpy as np
import torch.optim
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from framework.engine import get_optimizers
from framework.registry import Datasets, EvalFuncs
from models.AdaptorHelper import get_new_optimizers
from utils.tensor_utils import AverageMeterDict, zero_and_update
from framework.loss_and_acc import get_loss_and_acc
from utils.tensor_utils import to


@EvalFuncs.register('tta')
def TTA_eval_model(model, eval_data, lr, epoch, args, engine, mode):
    device = engine.device
    running_loss, running_corrects, shadow_running_corrects = AverageMeterDict(), AverageMeterDict(), AverageMeterDict()

    # State dicts
    cur_domain = Datasets[args.dataset].Domains[args.exp_num[0]]
    model_path = args.TTA_model_path
    # p = '{}/{}{}/models/model_best.pt'.format(model_path, cur_domain, engine.time)
    # p = f'ckpts/FSDCL/{cur_domain}0/models/model_best.pt'
    # p = '/data/zj/PycharmProjects/DG-Feature-Stylization-main/test—c/model/model-best.pth.tar'
    # p = '/data/gjt/sagnet-master/checkpoint/PACS/sagnet/cartoon,sketch,photo/checkpoint_latest.pth'
    if engine.target_domain[0] == 'art_painting':
        p = '/data/zj/PycharmProjects/FACT-main/output/PACS_ResNet50/art_painting/2022-05-29-14-06-46/best_model.tar'
    elif engine.target_domain[0] == 'cartoon':
        p = '/data/zj/PycharmProjects/FACT-main/output/PACS_ResNet50/cartoon/2022-05-29-14-06-40/best_model.tar'
    elif engine.target_domain[0] == 'photo':
        p = '/data/zj/PycharmProjects/FACT-main/output/PACS_ResNet50/photo/2022-05-29-13-59-08/best_model.tar'
    elif engine.target_domain[0] == 'sketch':
        p = '/data/zj/PycharmProjects/FACT-main/output/PACS_ResNet50/sketch/2022-05-29-14-06-52/best_model.tar'
    else:
        raise Exception("? {}".format(engine.target_domain[0]))
    # original_state_dicts = torch.load(p, map_location='cpu')['model']
    model.load_pretrained(p, absolute=True)
    # print("Loaded models from {}".format(model_path))

    changed = 0
    total = 0
    change_list = []
    sample_list = []
    label_list = []
    prob_list = []
    with torch.no_grad():
        model.eval()  # eval mode for normal test
        for i, data_list in enumerate(tqdm(eval_data)):
            test_data, test_label, aug_data = data_list['x'], data_list['label'], data_list['tta']
            test_data, aug_data, test_label = to([test_data, aug_data, test_label], device)

            outputs = model.step(test_data, test_label)
            logits = outputs['logits']
            _ = get_loss_and_acc(outputs, running_loss, running_corrects)

            # aug_data = aug_data.mean(1, keepdims=True)
            N, aug_n, C, H, W = aug_data.shape  #
            aug_data = aug_data.reshape(-1, C, H, W)
            aug_label = test_label.unsqueeze(1).repeat(aug_n, 1).reshape(-1)
            outputs2 = model.step(aug_data, aug_label)
            logits2 = outputs2['logits']
            # max_prob_mask = logits2.softmax(1).max(1)[0].view(N, aug_n, 1) > 0.95
            # mean_logits = (logits2.reshape(N, aug_n, -1) * max_prob_mask).sum(1) / max_prob_mask.sum(1)
            mean_logits = logits2.reshape(N, aug_n, -1).mean(1) + logits

            # second_pred = torch.topk(logits, 2, dim=-1)[1][:, 1]
            # idx = mean_logits.argmax(1) == logits.argmax(1)
            # second_pred[idx] = logits.argmax(1)[idx]
            # acc = (second_pred == test_label).sum() / len(second_pred)

            # w/ , w/o
            outputs2 = {'TTA': {'acc_type': 'acc', 'pred': mean_logits, 'target': test_label},}
            _ = get_loss_and_acc(outputs2, running_loss, running_corrects)

        #     previous_pred = logits.argmax(1)
        #     current_pred = logits2.argmax(1).view(N, aug_n)
        #     sample_list.append((logits.softmax(1).cpu().numpy(), logits2.reshape(N, aug_n, -1).softmax(2).mean(1).cpu().numpy()))
        #     label_list.append(test_label.cpu().numpy())
        #     for j, (p, c) in enumerate(zip(previous_pred, current_pred)):
        #         total += 1
        #         if len(c.unique()) > 1:
        #             changed += 1
        #             if len(logits) == 16:
        #                 pre_max_prob = logits.softmax(1).max(1)[0]
        #                 prob_list.append(pre_max_prob)
        #             change_list.append(i * 16 + j)
        # print(change_list)
        # print('Previous pred prob : ', torch.stack(prob_list).mean(), torch.stack(prob_list).std())
        # print('Total : {}, Changed : {}, PCR : {:.2f}'.format(total, changed, changed/total*100))
    # np.save('{}'.format(cur_domain), sample_list)
    # np.save('{}-l'.format(cur_domain), label_list)

    loss = running_loss.get_average_dicts()
    acc = running_corrects.get_average_dicts()  #
    if 'main' in acc:
        return acc['TTA'], (loss, acc)
    else:
        return 0, (loss, acc)


@torch.enable_grad()
def finetune_entropy(model, img, optimizer, k=1):
    avg_loss = []
    for i in range(k):
        logits = model.step(img, None)['logits']
        entropy_loss = - (logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
        gt_loss = 0

        prob = logits.softmax(1)
        max_prob = prob.max(1)[0] > 0.9
        peudo_label = prob.max(1)[1]
        low_loss = - (logits.softmax(1) * logits.log_softmax(1)).sum(1)[~max_prob].mean()
        high_loss = F.cross_entropy(logits, peudo_label, reduction='none')[max_prob].mean()

        # .mean(0)
        # prob = logits.softmax(1)
        # consistency_loss = ((prob.unsqueeze(1) - prob.unsqueeze(0)) ** 2).mean()
        loss = entropy_loss  # + consistency_loss
        zero_and_update(optimizer, loss)
        avg_loss.append(loss.distance_dict())


@torch.enable_grad()
def finetune_sep(model, test_data, logits, label, bn_optimizers, all_optimizers, k=3, threshold=0.95):
    optimizers = all_optimizers
    avg_loss = []
    conf = logits.softmax(1).max(1)[0] > threshold
    confident_label = logits.argmax(1)[conf]
    for i in range(k):
        logits = model(test_data, None)['logits']
        unconfident_logits = logits
        entropy_loss = - (unconfident_logits.softmax(1) * unconfident_logits.log_softmax(1)).sum(1).mean(0)

        confident_logits, unconfident_logits = logits[conf], logits[~conf]
        sup_loss = F.cross_entropy(confident_logits, confident_label)
        #
        entropy_loss.backward(retain_graph=True)
        for n, p in model.named_parameters():
            if 'bn' not in n:
                p.grad = None

        # loss = F.cross_entropy(logits, label)
        loss = entropy_loss
        [o.zero_grad() for o in optimizers]
        loss.backward()
        [o.step() for o in optimizers]
        avg_loss.append(loss.distance_dict())
    return logits

