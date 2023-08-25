import shutil
import inspect
import numpy as np
from copy import deepcopy
from pathlib import Path

from framework.registry import Models, Datasets, EvalFuncs, show_entries_and_files, TrainFuncs
from framework.engine import GenericEngine
from framework.log import generate_exp
from utils.tensor_utils import mkdir
import torch

torch.manual_seed(200423)
np.random.seed(200243)


def get_mean_dict(dict_list):
    keys = dict_list[0].keys()
    mean_dict = {}
    for k in keys:
        mean_dict[k] = np.mean([d[k] for d in dict_list])
    return mean_dict


class Experiments(object):
    def __init__(self, args):
        print(args)
        self.args = args
        self.exp_num = self.args.exp_num
        self.start_time = self.args.start_time
        self.times = self.args.times
        self.dataset = self.args.dataset
        self.domains = Datasets[self.dataset].Domains
        self.save_path = Path(self.args.save_path)
        mkdir(self.save_path, level=3)
        with open(self.save_path / 'args.txt', 'a') as f:
            f.write(str(args) + '\n\n')
        if args.show_entry:
            show_entries_and_files()

    def backup(self, args):
        print('Backing up..............')
        dir = self.save_path / 'backup'
        dir.mkdir(exist_ok=True)
        file = Models.get_src_file(args.model)
        print(file)
        shutil.copy(file, dir)

        file = TrainFuncs.get_src_file(args.train)
        print(file)
        shutil.copy(file, dir)

        file = EvalFuncs.get_src_file(args.eval)
        print(file)
        shutil.copy(file, dir)

        from dataloader.augmentations import RandAugment
        shutil.copy(inspect.getfile(RandAugment), dir)

        shutil.copy(Datasets.get_src_file(args.dataset), dir)

    def run(self):
        print()
        try:
            if self.args.do_train:
                self.backup(self.args)
            if self.exp_num[0] == -2:
                print('Run All Exp Many Times !!!')
                self.run_all_exp_many_times(self.times)
            elif self.exp_num[0] == -1:
                print('Run All Exp !!!')
                self.run_all_exp(self.start_time)
            else:
                for num in self.exp_num:
                    print('Run One Exp !!!')
                    assert num >= 0
                    self.run_one_exp(exp_idx=num, time=self.args.start_time)
        except Exception as e:
            import traceback
            traceback.print_exc()
            with open(self.save_path / 'error.txt', 'w') as f:
                traceback.print_exc(None, f)

    def run_all_exp_many_times(self, times=3):
        acc_array, acc_dict_array = [], []
        with open(self.save_path / 'many_exp.txt', 'a') as f:

            for t in range(times):
                print('============= Run {} ============='.format(self.start_time + t))
                acc_list, (keys, acc_dict_alist) = self.run_all_exp(self.start_time + t)
                acc_array.append(acc_list)
                acc_dict_array.append(acc_dict_alist)

            acc_array = np.array(acc_array)  # times x (domains+1)  # +1 is the mean acc
            acc_dict_array = np.array(acc_dict_array)  # times x (domains+1) x values  # +1 is the mean acc

            assert acc_array.shape[1] == len(self.domains) + 1
            std = acc_array.std(axis=0)
            mean = acc_array.mean(axis=0)

            dict_std = {k: acc_dict_array.std(0)[:, i] for i, k in enumerate(keys)}
            dict_mean = {k: acc_dict_array.mean(0)[:, i] for i, k in enumerate(keys)}

            names = self.domains + ['Mean']
            for i, (d, m, s) in enumerate(zip(names, mean, std)):
                print('{} : {:.2f}+-{:.2f}'.format(d, m, s))
                f.write('{} : {:.2f}+-{:.2f}\n'.format(d, m, s))
            print()

            for k in keys:
                print(k)
                for i, (d, m, s) in enumerate(zip(names, dict_mean[k], dict_std[k])):
                    print('{} : {:.2f}+-{:.2f}'.format(d, m, s))
                print()

            if self.args.do_train:
                # generate_last_exp(self.save_path, domains=self.domains, times=times)
                generate_exp(self.save_path, domains=self.domains, times=times, type='last')

    def run_all_exp(self, time):
        test_acc_list, test_acc_dict_list = [], []
        with open(str(self.save_path / 'all_exp.txt'), 'a') as f:
            print('------------- New Exp -------------')
            f.write('------------- New Exp -------------\n')
            for i, d in enumerate(self.domains):
                acc, acc_dict = self.run_one_exp(exp_idx=i, time=time)
                print(f'{d} : {acc:.2f}  [' + ', '.join([f'{k} : {v:.4f}' for k, v in acc_dict.items()]) + ']\n')
                f.write(f'{d} : {acc:.2f}  [' + ', '.join([f'{k} : {v:.4f}' for k, v in acc_dict.items()]) + ']\n')
                test_acc_list.append(acc * 100)
                test_acc_dict_list.append({k:v * 100 for k,v in acc_dict.items()})
            mean_acc = np.mean(test_acc_list)
            mean_acc_dict = get_mean_dict(test_acc_dict_list)
            print(f'Mean {mean_acc:.2f}  [' + ', '.join([f'{k} : {v:.4f}' for k, v in mean_acc_dict.items()]) + ']\n')
            f.write(f'{mean_acc:.2f}  [' + ', '.join([f'{k} : {v:.4f}' for k, v in mean_acc_dict.items()]) + ']\n')
            # print('Mean acc : {} {}\n\n'.format(mean_acc, mean_acc_dict))
            # f.write('Mean acc : {} {}\n\n'.format(mean_acc, mean_acc_dict))
        test_acc_list.append(mean_acc)
        test_acc_dict_list.append(mean_acc_dict)
        keys = test_acc_dict_list[0].keys()
        return test_acc_list, (keys, [[test_acc_dict_list[i][k] for k in keys] for i in range(len(test_acc_list))])

    def run_one_exp(self, exp_idx=0, time=0):
        args = deepcopy(self.args)
        args.exp_num = [exp_idx]
        args.save_path = str(Path(args.save_path) / '{}{}'.format(self.domains[exp_idx], time))
        engine = GenericEngine(args, time)
        test_acc, test_acc_dict = engine.train()
        return test_acc, test_acc_dict
