import time
from pathlib import Path
import numpy as np
import re
from tensorboardX import SummaryWriter
import os

from framework.registry import Datasets


class MyLogger:
    def __init__(self, args, root, target_domain, enable_tensorboard=False,
                 source_train_path='source_train.txt', source_eval_path='source_eval.txt', target_test_path='target_test.txt'):
        self.args = args
        root = Path(root)

        src_train_path = root / source_train_path
        src_eval_path = root / source_eval_path
        target_test_path = root / target_test_path
        self.paths = {
            'train': src_train_path,
            'eval': src_eval_path,
            'test': target_test_path,
            'EMA test': target_test_path
        }
        self.target_domain = target_domain
        self.enable_tensorboard = enable_tensorboard
        if self.enable_tensorboard:
            self.writer = SummaryWriter(str(root / 'tensorboard'), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S'))

    def log_seed(self, seed):
        with open(self.paths['test'], 'a') as f:
            f.write('Current Seed {}\n'.format(seed))

    def log_best(self, epoch, best_val_acc, loss, acc):
        with open(self.paths['test'], 'a') as f:
            f.write('Best test (domain : {}, bs : {}, val : {:.4f}): '.format(self.target_domain, self.args.batch_size, best_val_acc))
        print('Best test (domain : {}, bs : {}, val : {:.4f}): '.format(self.target_domain, self.args.batch_size, best_val_acc), end=' ')
        self.log_file('test', epoch, loss, acc)

    def log_str(self, mode, log):
        file = self.paths[mode]
        with open(file, 'a') as f:
            f.write(log + '\n')

    def log(self, mode, epoch, loss_dict, acc_dict):
        self.log_file(mode, epoch, loss_dict, acc_dict)
        if self.enable_tensorboard:
            self.tf_log_file(mode, epoch, loss_dict, acc_dict)

    def log_file(self, mode, epoch, loss_dict, acc_dict):
        loss_str = ''.join([' {}: {:.4f}'.format(k, v) for k, v in loss_dict.items()])
        acc_str = ''.join([' {}: {:.4f}'.format(k, v) if isinstance(v, (float, int)) else ' {}: {}'.format(k, v) for k, v in acc_dict.items()])
        t = time.strftime('[%Y-%m-%d %H:%M:%S]')
        log = '{} {:5s}: Epoch: {},   Loss : {} \t Acc : {}'.format(t, mode, epoch, loss_str, acc_str)
        print(log)
        for k in self.paths.keys():
            if k in mode:
                self.log_str(k, log)

    def tf_log_file(self, mode, epoch, loss_dict, acc_dict):
        assert self.enable_tensorboard, "Tensorboard not enabled!!!!"
        for k, v in loss_dict.items():
            self.writer.add_scalar('loss/{}/{}/{}'.format(self.target_domain, mode, k), v, epoch)
        for k, v in acc_dict.items():
            self.writer.add_scalar('acc/{}/{}/{}'.format(self.target_domain, mode, k), v, epoch)
        self.writer.flush()

    def log_output(self, log_dicts, iter, parent=''):
        for k, v in log_dicts.items():
            if 'hist' in k:
                self.writer.add_histogram(k, v, iter)
            else:
                self.writer.add_scalar(parent + '{}/{}'.format(self.target_domain, k), v, iter)
        self.writer.flush()


def generate_many_exp(path):
    import re
    p = Path(path)
    acc_dict = {}
    with open(str(p / 'all_exp.txt'), 'r') as f:
        for line in f:
            res = re.match(r'(.*) : (.*)', line)
            if res is None:
                continue
            domain, acc = res.groups()
            acc = float(acc)
            if domain in acc_dict:
                acc_dict[domain].append(acc)
            else:
                acc_dict[domain] = [acc]

    with open(p / 'many_exp.txt', 'a') as f:
        for d, acc_list in acc_dict.items():
            m = np.mean(acc_list)
            s = np.std(acc_list)
            print('{} : {:.2f}+-{:.2f} \n'.format(d, m, s))
            f.write('{} : {:.2f}+-{:.2f} \n'.format(d, m, s))
        f.write('\n')


def get_acc_list(folder, line_idx, times, domains):
    folder = Path(folder)
    # times x domains
    accs_list = []
    for i in range(times):
        accs = []
        for d in domains:
            file = folder / (d + str(i)) / 'target_test.txt'
            with open(str(file), 'r') as f:
                last_line = f.readlines()[line_idx]
                last_acc = float(re.match('.* Acc :.*main: (.*)', last_line).groups()[0][:6])
                accs.append(last_acc)
        accs_list.append(accs)
    return accs_list


def generate_exp(folder, domains, times=5, type='all', with_ema=False):
    if type == 'last':
        many_exp_file = 'last_many_exp.txt'
        all_exp_file = 'last_all_exp.txt'
        if with_ema:
            line_idx = -3
        else:
            line_idx = -2
    elif type == 'all':
        many_exp_file = 'many_exp.txt'
        all_exp_file = 'all_exp.txt'
        line_idx = -1
    elif type == 'ema':
        many_exp_file = 'ema_many_exp.txt'
        all_exp_file = 'ema_all_exp.txt'
        line_idx = -2

    import numpy as np
    folder = Path(folder)
    accs_list = get_acc_list(folder, line_idx, times, domains)
    accs_list = np.array(accs_list) * 100
    mean = accs_list.mean(0)
    std = accs_list.std(0)
    total_mean = accs_list.mean()
    total_std = accs_list.mean(1).std()

    with open(str(folder / all_exp_file), 'w') as f:
        for acc in accs_list:
            f.write('------------- New Exp -------------\n')
            for d, ac in zip(domains, acc):
                line = '{} : {:.2f}\n'.format(d, ac)
                f.write(line)
            line = '{} : {:.2f}\n'.format('Mean acc', acc.mean())
            f.write(line)
            f.write('\n')

    with open(str(folder / many_exp_file), 'w') as f:
        for d, m, s in zip(domains, mean, std):
            line = '{} : {:.2f}+-{:.2f}\n'.format(d, m, s)
            print(line)
            f.write(line)
        line = '{} : {:.2f}+-{:.2f}\n'.format('Mean', total_mean, total_std)
        print(line)
        f.write(line)
    print('Finished {}'.format(folder))


def delete(path):
    path = Path(path)
    for folder in path.iterdir():
        if folder.is_dir() and 'back' not in folder.name:
            for model in (folder / 'models').iterdir():
                # if models.name[-4] in '0123456789':
                os.remove(str(model))
                print(model)


def plot(keys, acc_list, title):
    lengths = [len(acc) for acc in acc_list]
    max_len = np.max(lengths)
    start = [max_len - l for l in lengths]
    for i in range(len(keys)):
        plt.plot(range(start[i], lengths[i] + start[i]), acc_list[i], label=keys[i])
    plt.legend()
    plt.title(title)
    plt.show()


def get_last_mean_acc(domain_acc_list, keys):
    # domain -> task -> acc_list
    res = []
    for d in domain_acc_list:
        res.append([acc[-1] for acc in d])
    # domain x task
    res = np.array(res)
    mean_acc = np.mean(res, 0)
    print(dict(zip(keys, mean_acc)))


def read_file(file, first_keys=None):
    if first_keys is None:
        first_keys = ['^\[.*\] .*main: (.*)']
    with open(file, 'r') as f:
        lines = f.readlines()[:-1]
    all_acc_list = []
    for k in first_keys:
        acc_list = []
        for line in lines:
            acc = re.match(k, line)
            if acc is not None:
                acc_list.append(float(acc.groups()[0]))
        all_acc_list.append(acc_list)
    return all_acc_list


def read_acc(folder):
    folder = Path(folder)
    domains = Datasets['PACS'].Domains
    times = 5
    for t in range(times):
        domain_acc_list = []
        for d in domains:
            first_keys = ['test'] + ['EMA{}'.format(i) for i in range(5)]
            p = folder / (d + str(t)) / 'target_test.txt'
            all_acc_list = read_file(p, first_keys)
            domain_acc_list.append(all_acc_list)
        get_last_mean_acc(domain_acc_list, first_keys)


if __name__ == '__main__':
    domains = Datasets['MDN'].Domains
    # for f in Path('/data/zj/PycharmProjects/TTA/NEW').iterdir():
    #     if 'test' not in str(f.absolute()):
    #         generate_exp(f, domains, times=5, type='last', )
    # for folder in Path('/data/zj/PycharmProjects/DomainAdaptation/script/FOMAML_New').iterdir():
    #     delete(folder)
    generate_exp('/data/zj/PycharmProjects/TTA/AdaBN/meta_norm_MDN', domains, times=3, type='all')
