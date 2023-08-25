import copy
import errno
import functools
import os
import signal
import types
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.loss_and_acc import get_loss_and_acc
from utils.tensor_utils import to, zero_and_update


def put_theta(model, theta):
    if theta is None:
        return model

    def k_param_fn(tmp_model, name=None):
        if len(theta) == 0:
            return

        if len(tmp_model._modules) != 0:
            for (k, v) in tmp_model._modules.items():
                if name == '':
                    k_param_fn(v, name=str(k))
                else:
                    k_param_fn(v, name=str(name + '.' + k))

        # WARN : running_mean, 和 running_var 不是 parameter，所以在 new 中不会被更新
        for (k, v) in tmp_model._parameters.items():
            if isinstance(v, torch.Tensor) and str(name + '.' + k) in theta.keys():
                tmp_model._parameters[k] = theta[str(name + '.' + k)]
            # else:
            #     print(name+'.'+k)
            # theta.pop(str(name + '.' + k))

        for (k, v) in tmp_model._buffers.items():
            if isinstance(v, torch.Tensor) and str(name + '.' + k) in theta.keys():
                tmp_model._buffers[k] = theta[str(name + '.' + k)]
            # else:
            #     print(k)
            # theta.pop(str(name + '.' + k))

    k_param_fn(model, name='')
    return model


def get_parameters(model):
    # note : you can direct manipulate these data reference which is related to the original models
    parameters = dict(model.named_parameters())
    states = dict(model.named_buffers())
    return parameters, states


def put_parameters(model, param, state):
    model = put_theta(model, param)
    model = put_theta(model, state)
    return model


def update_parameters(loss, names_weights_dict, lr, use_second_order, retain_graph=True, grads=None, ignore_keys=None):
    def contains(key, target_keys):
        if isinstance(target_keys, (tuple, list)):
            for k in target_keys:
                if k in key:
                    return True
        else:
            return key in target_keys

    new_dict = {}
    for name, p in names_weights_dict.items():
        if p.requires_grad:
            new_dict[name] = p
        # else:
        #     print(name)
    names_weights_dict = new_dict

    if grads is None:
        grads = torch.autograd.grad(loss, names_weights_dict.values(), create_graph=use_second_order, retain_graph=retain_graph, allow_unused=True)
    names_grads_wrt_params_dict = dict(zip(names_weights_dict.keys(), grads))
    updated_names_weights_dict = dict()

    for key in names_grads_wrt_params_dict.keys():
        if names_grads_wrt_params_dict[key] is None:
            continue  # keep the original state unchanged

        if ignore_keys is not None and contains(key, ignore_keys):
            # print(f'ignore {key}' )
            continue

        updated_names_weights_dict[key] = names_weights_dict[key] - lr * names_grads_wrt_params_dict[key]
    return updated_names_weights_dict


def cat_meta_data(data_list):
    new_data = {}
    for k in data_list[0].keys():
        l = []
        for data in data_list:
            l.append(data[k])
        new_data[k] = torch.cat(l, 0)
    return new_data


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


# @timeout(3)
def get_image_and_label(loaders, idx_list, device):
    if not isinstance(idx_list, (list, tuple)):
        idx_list = [idx_list]

    data_lists = []
    for i in idx_list:
        data = loaders[i].next()
        data = to(data, device)  # , non_blocking=True)
        # data = loaders[i].next()
        data_lists.append(data)
    return cat_meta_data(data_lists)


def split_image_and_label(data, size, loo=False):
    n_domains = list(data.values())[0].shape[0] // size
    idx_sequence = list(np.random.permutation(n_domains))
    if loo:
        n_domains = 2
    res = [{} for _ in range(n_domains)]

    for k, v in data.items():
        split_data = torch.split(v, size)
        if loo:  # meta_train, meta_test
            res[0][k] = torch.cat([split_data[_] for _ in idx_sequence[:len(split_data) - 1]])
            res[1][k] = split_data[idx_sequence[-1]]
        else:
            for i, d in enumerate(split_data):
                res[i][k] = d
    return res


def new_split_image_and_label(data, size, loo=False):
    n_domains = list(data.values())[0].shape[0] // size
    if loo:
        n_domains = 2
    res = [{} for _ in range(n_domains)]

    for k, v in data.items():
        split_data = torch.split(v, size)
        if loo:  # meta_train, meta_test
            res[0][k] = torch.cat(split_data[:2])
            res[1][k] = torch.cat(split_data[2:])
        else:
            for i, d in enumerate(split_data):
                res[i][k] = d
    return res


def init_network(meta_model, meta_lr, previous_opt=None, momentum=0.9, Adam=False, beta1=0.9, beta2=0.999, device=None):
    fast_model = copy.deepcopy(meta_model).train()
    if device is not None:
        fast_model.to(device)
    if Adam:
        fast_opts = torch.optim.Adam(fast_model.parameters(), lr=meta_lr, betas=(beta1, beta2), weight_decay=5e-4)
    else:
        fast_opts = torch.optim.SGD(fast_model.parameters(), lr=meta_lr, weight_decay=5e-4, momentum=momentum)

    if previous_opt is not None:
        fast_opts.load_state_dict(previous_opt.state_dict())
    return fast_model, fast_opts


def load_state(new_opts, old_opts):
    [old.load_state_dict(new.state_dict()) for old, new in zip(old_opts, new_opts)]


def update_meta_model(meta_model, fast_param_list, optimizers, meta_lr=1):
    meta_params, meta_states = get_parameters(meta_model)

    optimizers.zero_grad()

    # update grad
    for k in meta_params.keys():
        new_v, old_v = 0, meta_params[k]
        for m in fast_param_list:
            new_v += m[0][k]
        new_v = new_v / len(fast_param_list)
        meta_params[k].grad = ((old_v - new_v) / meta_lr).data
    optimizers.step()


def avg_meta_model(meta_model, fast_param_list):
    meta_params, meta_states = get_parameters(meta_model)

    # update grad
    for k in meta_params.keys():
        new_v, old_v = 0, meta_params[k]
        for m in fast_param_list:
            new_v += m[k]
        new_v = new_v / len(fast_param_list)
        meta_params[k].data = new_v.data


def add_grad(meta_model, fast_model, factor):
    meta_params, meta_states = get_parameters(meta_model)
    fast_params, fast_states = get_parameters(fast_model)
    grads = []
    for k in meta_params.keys():
        new_v, old_v = fast_params[k], meta_params[k]
        if meta_params[k].grad is None:
            meta_params[k].grad = ((old_v - new_v) * factor).data  # if data is not used, the tensor will cause the memory leak
        else:
            meta_params[k].grad += ((old_v - new_v) * factor).data  # if data is not used, the tensor will cause the memory leak
        grads.append((old_v - new_v))
    return grads


def compare_two_dicts(d1, d2):
    flag = True
    for k in d1.keys():
        if not ((d1[k] - d2[k]).abs().max() < 1e-7):
            print(k, (d1[k] - d2[k]).abs().max())
            flag = False
    return flag


def inner_loop(meta_model, meta_train_data, meta_test_data, steps, meta_lr, opt_states, running_loss, running_corrects,
               meta_test=True, train_aug=False, test_aug=False):
    fast_model = copy.deepcopy(meta_model).train()
    put_parameters(fast_model, None, get_parameters(meta_model)[1])  # Only Put BN for fair comparison
    fast_opts = torch.optim.SGD(fast_model.parameters(), lr=meta_lr, weight_decay=5e-4, momentum=0.9)
    if opt_states is not None:
        fast_opts.load_state_dict(opt_states)

    # meta train
    if not train_aug:
        meta_train_data = copy.deepcopy(meta_train_data)
        meta_train_data['aug_x'] = None
    for i in range(steps):
        out = fast_model(**meta_train_data, meta_train=True, do_aug=train_aug)
        meta_train_loss2 = get_loss_and_acc(out, None, None)
        zero_and_update([fast_opts], meta_train_loss2)

    if meta_test:
        # meta test
        if not test_aug:
            meta_test_data = copy.deepcopy(meta_test_data)
            meta_test_data['aug_x'] = None
        out = fast_model(**meta_test_data, meta_train=False, do_aug=test_aug)
        meta_val_loss2 = get_loss_and_acc(out, running_loss, running_corrects)
        zero_and_update(fast_opts, meta_val_loss2)
        return fast_model, fast_opts.state_dict(), out
    else:
        return fast_model, fast_opts


def correlation(grad1, grad2, cos=False):
    all_sim = []
    for g1, g2 in zip(grad1, grad2):
        if cos:
            sim = F.cosine_similarity(g1.view(-1), g2.view(-1), 0)
        else:
            sim = (g1 * g2).sum()
        all_sim.append(sim)
    all_sim = torch.stack(all_sim)
    return all_sim.mean()


def regularize_params(meta_model, params, opts, weight):
    def get_direction(param1, param2):
        dirs = []
        for p1, p2 in zip(param1, param2):
            dirs.append(p2 - p1)
        return dirs

    def get_mean(dirs):
        mean_dir = []
        for ls in zip(*dirs):
            v = 0
            for m in ls:
                v += m
            v = v / len(ls)
            mean_dir.append(v)
        return mean_dir

    meta_param = get_parameters(meta_model)[0]

    # get gradient direction from each models
    dirs = [get_direction(meta_param.values(), param.values()) for param in params]

    # get mean gradient direction
    mean_dir = get_mean(dirs)

    # measure distance between mean and other directions
    dists = []
    for i in range(len(dirs)):
        for j in range(len(dirs)):
            if j > i:
                dists.append(correlation(dirs[i], dirs[j], cos=True))
    dists = 1 - torch.stack(dists).mean()
    zero_and_update(opts, dists * weight)  # w/o, w/
    return dists


def mixup_parameters(params, num=2, alpha=1):
    assert num <= len(params)
    selected_list = np.random.permutation(len(params))[:num]
    if alpha > 0:
        ws = np.float32(np.random.dirichlet([alpha] * num))  # Random mixup params
    else:
        ws = [1 / num] * num  # simply average model
    params = [params[i] for i in selected_list]
    new_param = {}
    for name in params[0].keys():
        new_p = 0
        for w, p in zip(ws, params):
            new_p += w * p[name]
        new_param[name] = new_p
    return new_param, selected_list


def average_models(models):
    params = [get_parameters(m)[0] for m in models]
    new_param, _ = mixup_parameters(params, num=len(params), alpha=0)
    new_model = copy.deepcopy(models[0])
    averaged_model = put_parameters(new_model, new_param, None)
    return averaged_model


def get_consistency_loss(logits_clean, logits_aug=None, T=4, weight=2):
    if logits_aug is None:
        length = len(logits_clean)
        logits_clean, logits_aug = logits_clean[length // 2:], logits_clean[:length // 2]
    logits_clean = logits_clean.detach()
    p_clean, p_aug = (logits_clean / T).softmax(1), (logits_aug / T).softmax(1)
    p_mixture = ((p_aug + p_clean) / 2).clamp(min=1e-7, max=1).log()
    loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') + F.kl_div(p_mixture, p_aug, reduction='batchmean')) * weight
    return loss


class AveragedModel(nn.Module):
    def __init__(self, start_epoch=0, device=None, lam=None, avg_fn=None):
        super(AveragedModel, self).__init__()
        self.device, self.start_epoch = device, start_epoch
        self.module = None
        self.lam = lam
        self.register_buffer('n_averaged', torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, lamd):
                return lamd * averaged_model_parameter + (1 - lamd) * model_parameter
        self.avg_fn = avg_fn

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.module.step(*args, **kwargs)

    def init_model(self, model, epoch):
        if self.module is None:
            self.module = copy.deepcopy(model)
            if self.device is not None:
                self.module = self.module.to(self.device)

    def update_parameters(self, model, epoch):
        if epoch < self.start_epoch:
            return

        if self.module is None:
            self.module = copy.deepcopy(model)
            if self.device is not None:
                self.module = self.module.to(self.device)
            return

        if self.lam is None:
            lam = self.n_averaged.to(self.device) / (self.n_averaged.to(self.device) + 1)
        else:
            lam = self.lam
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_, lam))
        self.n_averaged += 1

    def update_bn(self, loader, epoch, iters=None, model=None, meta=False):
        model = self.module if model is None else model
        if epoch < self.start_epoch:
            return
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum

        if not momenta:
            return

        was_training = model.training
        model.train()
        for module in momenta.keys():
            module.momentum = None
            module.num_batches_tracked *= 0

        if meta:
            with torch.no_grad():
                inner_loops = len(loader) if iters is None else iters
                for i in range(inner_loops):
                    data_list = get_image_and_label(loader, [0, 1, 2], device=self.device)
                    model.step(**data_list)
        else:
            with torch.no_grad():
                inner_loops = len(loader) if iters is None else iters
                loader = iter(loader)
                for i in range(inner_loops):
                    data_list = to(next(loader), self.device)
                    model.step(**data_list)

        for bn_module in momenta.keys():
            bn_module.momentum = momenta[bn_module]
        model.train(was_training)


def freeze(model, name, freeze, reverse=False):
    for n, p in model.named_parameters():
        if not reverse:
            if name in n:
                p.requires_grad = freeze
        else:
            if name not in n:
                p.requires_grad = freeze


@contextmanager
def meta_learning_MAML(meta_model):
    fast_model = copy.deepcopy(meta_model)
    params, states = get_parameters(meta_model)
    fast_model = put_parameters(fast_model, params, states).train()

    def meta_step(self, meta_loss, meta_lr, use_second_order=False, ignore_keys=None):
        params = get_parameters(self)[0]
        params = update_parameters(meta_loss, params, meta_lr, use_second_order=use_second_order, ignore_keys=ignore_keys)
        put_parameters(self, params, None)

    fast_model.meta_step = types.MethodType(meta_step, fast_model)  # assign method to the instance
    yield fast_model
    del fast_model, params, states

