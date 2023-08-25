import os
import random

import torch
import numpy as np
from torch.optim import SGD, Adam
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from framework.log import MyLogger
from framework.registry import Models, Datasets, Schedulers, TrainFuncs, EvalFuncs


def extract_parameters(model, lr):
    if isinstance(model, (list, tuple)):
        if isinstance(model[0], nn.Parameter):
            params = [{'params': param, 'lr': lr} for param in model]
        elif isinstance(model[0], nn.Module):
            params = [{'params': model_.parameters(), 'lr': lr} for model_ in model]
        else:
            raise Exception("Unkown models {}".format(type(model)))
    else:
        if isinstance(model, nn.Parameter):
            params = [{'params': model, 'lr': lr}]
        else:
            params = [{'params': model.parameters(), 'lr': lr}]
    return params


def get_optimizers(model, args):
    init_lr, fc_weight = args.lr, args.fc_weight
    param_lr_lists = model.get_lr(fc_weight) if hasattr(model, 'get_lr') else [(model, fc_weight)]

    if args.opt_split:
        optimizer = []
        for model, weight in param_lr_lists:
            params = extract_parameters(model, init_lr * weight)
            if args.optimizer.lower() == 'sgd':
                opt = SGD(params, lr=init_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
            elif args.optimizer.lower() == 'adam':
                opt = Adam(params, lr=init_lr, betas=(args.beta1, args.beta2), weight_decay=5e-4)  # , amsgrad=True)
            else:
                raise Exception("Unknown optimizer : {}".format(args.optimizer))
            optimizer.append(opt)
    else:
        params = []
        for model, weight in param_lr_lists:
            params.extend(extract_parameters(model, init_lr * weight))
        if args.optimizer.lower() == 'sgd':
            optimizer = SGD(params, lr=init_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer.lower() == 'adam':
            optimizer = Adam(params, lr=init_lr, betas=(args.beta1, args.beta2), weight_decay=5e-4)  # , amsgrad=True)
        else:
            raise Exception("Unknown optimizer : {}".format(args.optimizer))
    return optimizer


def get_scheduler(args, opt):
    num_epoch = args.num_epoch
    if args.dataset == 'digits_dg':
        lr_step = 20
    else:
        lr_step = args.num_epoch * .8
    if args.scheduler == 'inv':
        schedulers = Schedulers[args.scheduler](optimizer=opt, alpha=10, beta=0.75, total_epoch=num_epoch)
    elif args.scheduler == 'step':
        schedulers = Schedulers[args.scheduler](optimizer=opt, step_size=lr_step, gamma=args.lr_decay_gamma)
    elif args.scheduler == 'cosine':
        schedulers = Schedulers[args.scheduler](optimizer=opt, T_max=10, eta_min=1e-5)
    else:
        raise ValueError('Name of scheduler unknown %s' % args.scheduler)
    return num_epoch, schedulers


class GenericEngine(object):
    def __init__(self, args, time):
        self.set_seed(time*10000)
        self.args = args
        self.time = time
        self.args.time = time

        self.path = Path(args.save_path)
        print(f'Save path : {self.path.absolute()}')
        self.path.mkdir(exist_ok=True)
        (self.path / 'models').mkdir(exist_ok=True)

        self.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        data_config = Datasets[args.dataset](args)
        self.num_classes = data_config.NumClasses
        self.model = Models[args.model](num_classes=self.num_classes, pretrained=True, args=args).to(self.device)
        (self.source_train, self.source_val, self.target_test), self.target_domain = self.get_loaders()
        self.optimizers = get_optimizers(self.model, args)
        self.num_epoch, self.schedulers = get_scheduler(args, self.optimizers)
        self.logger = MyLogger(args, self.path, '_'.join(self.target_domain))

        self.global_parameters = {}

        if len(args.load_path) > 0:
            self.load_model(args.load_path)

    def get_loaders(self):
        data_config = Datasets[self.args.dataset](self.args)
        return data_config.get_loaders(self.model.get_aug_funcs() if hasattr(self.model, 'get_aug_funcs') else None), data_config.target_domains

    def set_seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self):
        best_acc, test_acc, best_epoch = 0, 0, 0

        if self.args.do_train:
            for epoch in tqdm(range(self.num_epoch)):
                lr = self.optimizers[0].param_groups[0]['lr'] if isinstance(self.optimizers, (list, tuple)) else self.optimizers.param_groups[0]['lr']
                print('Epoch: {}/{}, Lr: {:.6f}'.format(epoch, self.num_epoch - 1, lr))
                print('Temporary Best Accuracy is {:.4f} ({:.4f} at Epoch {})'.format(test_acc, best_acc, best_epoch))

                (loss_dict, acc_dict) = TrainFuncs[self.args.train](self.model, self.source_train, lr, epoch, self.args, self, mode='train')
                self.logger.log('train', epoch, loss_dict, acc_dict)

                if epoch % self.args.eval_step == 0:
                    acc, (loss_dict, acc_dict) = EvalFuncs[self.args.eval](self.model, self.source_val, lr, epoch, self.args, self, mode='eval')
                    self.logger.log('eval', epoch, loss_dict, acc_dict)

                    acc_, (loss_dict, acc_dict) = EvalFuncs[self.args.eval](self.model, self.target_test, lr, epoch, self.args, self, mode='test')
                    self.logger.log('test', epoch, loss_dict, acc_dict)

                if epoch > 0 and epoch % self.args.save_step == 0 and epoch >= self.args.start_save_epoch:
                    self.save_model(f'{epoch}.pt')

                if acc >= best_acc:
                    best_acc, test_acc, best_epoch = acc, acc_, epoch
                    self.save_model('model_best.pt')

                self.schedulers.step()

            if self.args.save_last:
                self.save_model('model_last.pt')

        if self.args.test_with_eval:
            print('Test with source validation set!!!')
            test_acc, test_acc_dict = self.test(best_epoch, best_acc, loader=self.source_val)
        else:
            test_acc, test_acc_dict = self.test(best_epoch, best_acc, loader=self.target_test)
        self.save_global_parameters()
        return test_acc, test_acc_dict

    def save_model(self, name='model_best.pt'):
        save_dict = {
            'model': self.model.state_dict(),
            'opt': self.optimizers.state_dict()
        }
        torch.save(save_dict, os.path.join(self.path, 'models', name))

    def load_model(self, path=None):
        if path is None:
            path = os.path.join(self.path, 'models', "model_best.pt")
        else:
            path = os.path.join(path, '_'.join(self.target_domain) + str(self.time), 'models', "model_best.pt")

        if os.path.exists(path):
            m = self.model.load_pretrained(path, absolute=True)
            if 'opt' in m:
                try:
                    ret1 = self.optimizers.load_state_dict(m['opt'])
                except Exception as e:
                    print(e)
                print('Load optimizer from {}'.format(path), ret1)
        else:
            print('Model in {}, Not found !!!!'.format(path))
        return self.model

    def test(self, best_epoch=0, best_acc=0, loader=None):
        lr = self.optimizers[0].param_groups[0]['lr'] if isinstance(self.optimizers, (list, tuple)) else self.optimizers.param_groups[0]['lr']
        self.load_model()
        self.model = self.model.to(self.device)
        test_acc, (loss_dict, acc_dict) = EvalFuncs[self.args.eval](self.model, loader, lr, best_epoch, self.args, self, mode='test')
        self.logger.log_best(best_epoch, best_acc, loss_dict, acc_dict)
        return test_acc, acc_dict

    def save_global_parameters(self):
        for k, v in self.global_parameters.items():
            np.save(self.target_domain[0]+'-'+k, v)
