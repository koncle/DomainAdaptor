import time
from pathlib import Path

import numpy as np
import torch

__all__ = ['to', 'to_numpy', 'cat', 'zero_and_update', 'mkdir', 'Timer', 'AverageMeter', 'AverageMeterDict']

"""
Tensor Utils
"""


def to(tensors, device, non_blocking=False):
    res = []
    if isinstance(tensors, (list, tuple)):
        for t in tensors:
            res.append(to(t, device, non_blocking=non_blocking))
        return res
    elif isinstance(tensors, (dict,)):
        res = {}
        for k, v in tensors.items():
            res[k] = to(v, device, non_blocking=non_blocking)
        return res
    else:
        if isinstance(tensors, torch.Tensor):
            return tensors.to(device, non_blocking=non_blocking)
        else:
            return tensors


def record_stream(tensors):
    if isinstance(tensors, (list, tuple)):
        for t in tensors:
            record_stream(t)
    elif isinstance(tensors, (dict,)):
        for k, v in tensors.items():
            record_stream(v)
    else:
        if isinstance(tensors, torch.Tensor):
            tensors.record_stream(torch.cuda.current_stream())


def to_numpy(tensor):
    if tensor is None:
        return None
    elif isinstance(tensor, (tuple, list)):
        res = []
        for t in tensor:
            res.append(to_numpy(t))
        return res
    else:
        if isinstance(tensor, np.ndarray) or str(type(tensor))[8:13] == 'numpy':
            return tensor
        else:
            return tensor.detach().cpu().numpy()


def cat(tensors, axis=0):
    res = []
    for t in tensors:
        if t is None:
            res.append(None)
        elif isinstance(t[0], torch.Tensor):
            res.append(torch.cat(t, dim=axis))
        else:
            res.append(np.concatenate(t, axis=axis))
    return res


def zero_and_update(optimizers, loss):
    if isinstance(optimizers, (list, tuple)):
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for opt in optimizers:
            opt.step()
    else:
        optimizers.zero_grad()
        loss.backward()
        optimizers.step()


"""
Output Utils
"""


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.n = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.min = 1e8
        self.max = -1e8
        # self.l = []
        return self

    def update(self, val, n=1):
        self.val = val
        self.n = n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        # self.l.append(val)
        return self


class AverageMeterDict(object):
    def __init__(self):
        self.dict = {}

    def update(self, name, val, n=1):
        if name not in self.dict:
            self.dict[name] = AverageMeter()
        self.dict[name].update(val, n)

    def update_dict(self, d, n=1):
        for name, val in d.items():
            if isinstance(val, (list, tuple)):
                continue
            self.update(name, val, n)

    def get_average_dicts(self):
        return {k: v.avg for k, v in self.dict.items()}

    def get_current_dicts(self):
        return {k: v.val / v.n for k, v in self.dict.items()}

    def print(self, current=True, end='\n'):
        strs = []
        d = self.get_current_dicts() if current else self.get_average_dicts()
        for k, v in d.items():
            strs.append('{}: {:.4f}'.format(k, v))
        strs = '{' + ', '.join(strs) + '} '
        print(strs, end=end)


"""
Other Utils
"""


def mkdir(path, level=2, create_self=True):
    """ Make directory for this path,
    level is how many parent folders should be created.
    create_self is whether create path(if it is a file, it should not be created)

    e.g. : mkdir('/home/parent1/parent2/folder', level=3, create_self=True),
    it will first create parent1, then parent2, then folder.

    :param path: string
    :param level: int
    :param create_self: True or False
    :return:
    """
    p = Path(path)
    if create_self:
        paths = [p]
    else:
        paths = []
    level -= 1
    while level != 0:
        p = p.parent
        paths.append(p)
        level -= 1

    for p in paths[::-1]:
        p.mkdir(exist_ok=True)


class Timer(object):
    def __init__(self, name='', thresh=0, verbose=True):
        self.start_time = time.time()
        self.verbose = verbose
        self.duration = 0
        self.thresh=thresh
        self.name = name

    def restart(self):
        self.duration = self.start_time = time.time()
        return self.duration

    def stop(self):
        time.asctime()
        return time.time() - self.start_time

    def get_last_duration(self):
        return self.duration

    def get_formatted_duration(self, duration=None):
        def sec2time(seconds):
            s = seconds % 60
            seconds = seconds // 60
            m = seconds % 60
            seconds = seconds // 60
            h = seconds % 60
            return h, m, s

        if duration is None:
            duration = self.duration
        return '{} Time {:^.0f} h, {:^.0f} m, {:^.4f} s'.format(self.name, *sec2time(duration))

    def __enter__(self):
        self.restart()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = self.stop()
        if self.verbose and self.duration > self.thresh:
            print(self.get_formatted_duration())

