import torch.nn.functional as F
import numpy as np
import torch
from torch import nn

from framework.registry import LossFuncs, AccFuncs, Datasets
from utils.tensor_utils import to_numpy
from utils.visualize import bar_plot


class ClassChangeMeasure():
    """
    Measure Class Changes
    """

    def __init__(self, classes):
        self.classes = classes
        self.total_changed = 0
        self.changed_to = np.array([0] * classes)
        self.should_change_to = np.array([0] * classes)
        self.changed_and_cor = np.array([0] * classes)

        self.changed_prob_dist = [0] * 10
        self.total_prob_dist = [0] * 10
        self.dist1 = DistributionMeasure()
        self.dist2 = DistributionMeasure()

    def process_changed(self, all_probs, label, mask, length=10):
        self.dist1(all_probs, label)
        self.dist2(all_probs[mask], label[mask])
        # max_prob = probs.max(1)[0].detach()
        # all_max_prob = all_probs.max(1)[0].detach()
        # interval = 1 / length
        # for i in range(length):
        #     low, high = interval * i, interval * (i + 1)
        #     cur_mask = (low <= max_prob) & (max_prob < high)
        #     self.changed_prob_dist[i] += cur_mask.sum().item()
        #     cur_mask = (low <= all_max_prob) & (all_max_prob < high)
        #     self.total_prob_dist[i] += cur_mask.sum().item()

    def process(self, pre_logits, cur_logits, label):
        cur_prob = cur_logits.softmax(1)
        pre_prob = pre_logits.softmax(1)
        mask = (pre_logits.argmax(1) != cur_logits.argmax(1))
        cur_correct_mask = (cur_logits.argmax(1) == label)

        if mask.sum() > 0:
            # self.process_changed(pre_prob[mask], pre_prob)
            self.process_changed(pre_logits, label, mask)

            self.total_changed += mask.sum().item()
            cur_pred_class = cur_prob.argmax(1)
            pre_sorted_classes = pre_prob.argsort(dim=-1, descending=True)

            for i in range(0, pre_logits.shape[1]):  # remove the same classes
                changed_to_i_mask = pre_sorted_classes[:, i] == cur_pred_class
                if i > 0:
                    self.changed_to[i] += changed_to_i_mask.sum().item()
                    self.changed_and_cor[i] += cur_correct_mask[changed_to_i_mask].sum().item()

                self.should_change_to[i] += (label == pre_sorted_classes[:, i])[mask].sum().item()

    def print(self):
        ratio = self.changed_and_cor / self.changed_to
        print(f'Total changed    : {self.total_changed}')
        print(f'changed to       : {self.changed_to}')
        print(f'correct          : {self.changed_and_cor}, ratio : {ratio}')
        print(f'Should change to : {self.should_change_to}')
        self.dist1.print()
        self.dist2.print()
        # print(f'Changed distribution : {self.changed_prob_dist}')
        # print(f'Total distribution : {self.total_prob_dist}')

    def bar_plot(self, title, save_name=False):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        colors = ['#5A9BD5', '#FF9966', '#1B9E78', '#ff585d', '#614ad3', '#feda77']
        width = 0.3
        length = 10
        x = np.array(list(range(len(self.changed_and_cor)))).astype(np.int)
        ax.bar(x[:length] - width / 1.7, self.changed_and_cor[:length], width, label='Correct', color=colors[0])  #
        ax.bar(x[:length] - width / 1.7, self.changed_to[:length] - self.changed_and_cor[:length], width,
               bottom=self.changed_and_cor[:length], label='Changed', color=colors[1])
        ax.bar(x[:length] + width / 1.7, self.should_change_to[:length], width, label='GT', color=colors[2])

        size = 20
        ax.set_ylabel('# of changed samples', size=size)
        ax.set_xlabel('Classes sorted by confidence', size=size)
        ax.set_title(title, size=size + 1)
        ax.set_xticks(list(range(0, self.classes))[:length], ['{:d}'.format(i) for i in range(1, self.classes + 1)][:length], size=size)
        ax.tick_params(axis='both', labelsize=size)
        ax.legend(fontsize=size)
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(save_name, bbox_inches='tight')
        plt.show()


class LogitsScaleMeasure():
    """
    Measure Probability distribution
    """

    def __init__(self, scales=None):
        self.prob_list = []
        self.scales = scales if scales is not None else range(1, 10)

    def clear(self):
        self.prob_list = []

    def __call__(self, logits):
        l = []
        for s in self.scales:
            prob = (logits / s).softmax(1)
            max_prob = prob.max(1)[0].mean().item()
            l.append(max_prob)
        self.prob_list.append(l)
        return l

    def get_averaged_prob(self):
        probs = np.stack(self.prob_list, 0).mean(0)
        return probs

    def print(self):
        prob = self.get_averaged_prob()
        print(prob)


class DistributionMeasure():
    """
    Measure Probability distribution
    """

    def __init__(self):
        self.count_list = []
        self.correct_list = []
        self.count = 0

    def clear(self):
        self.count_list = []

    def __call__(self, logits, label):
        prob = logits.softmax(1)
        max_prob = prob.max(1)[0].detach()
        cor = logits.argmax(1) == label
        length = 10
        interval = 1 / length
        counts, corrects = [], []
        for i in range(length):
            low, high = interval * i, interval * (i + 1)
            cur_mask = (low <= max_prob) & (max_prob < high)
            counts.append(cur_mask.sum().item())
            corrects.append((cur_mask & cor).sum().item())
        # count = torch.histc(max_prob, bins=10, min=0, max=1).numpy()
        self.count_list.append(counts)
        self.correct_list.append(corrects)
        self.count += 1
        return counts

    def get_averaged_corrects(self):
        counts = self.get_averaged_counts()
        corrects = np.stack(self.correct_list, 0).mean(0)
        averaged_corrects = corrects / counts
        averaged_corrects[np.isnan(averaged_corrects)] = 0
        return averaged_corrects

    def get_averaged_counts(self):
        count = np.stack(self.count_list, 0)
        return count.mean(0)

    def get_sum_counts(self):
        count = np.stack(self.count_list, 0)
        return count.sum(0)

    def print(self):
        corrects = self.get_averaged_corrects()
        print(corrects)

    def print_num(self):
        print(np.stack(self.count_list, 0).sum(0))

    def bar_plot(self):
        corrects = self.get_averaged_corrects()
        print(corrects)
        bar_plot([corrects], plot_text=True)


class LossDistributionMeasure(DistributionMeasure):
    def __init__(self):
        super(LossDistributionMeasure, self).__init__()
        self.loss_list = []

    def __call__(self, logits, losses):
        prob = logits.softmax(1)
        max_prob = prob.max(1)[0].detach()
        length = 10
        interval = 1 / length
        counts, loss_list = [], []
        for i in range(length):
            low, high = interval * i, interval * (i + 1)
            cur_mask = (low <= max_prob) & (max_prob < high)
            loss = losses[cur_mask].sum()
            loss_list.append(loss.item())
            counts.append(cur_mask.sum().item())
        self.count_list.append(counts)
        self.loss_list.append(loss_list)
        self.count += 1
        return losses

    def get_averaged_loss(self):
        counts = self.get_averaged_counts()
        losses = np.stack(self.loss_list, 0).mean(0)
        averaged_loss = losses / counts
        averaged_loss[np.isnan(averaged_loss)] = 0
        return averaged_loss

    def print(self):
        losses = self.get_averaged_loss()
        print(losses)


class ClassMeasure():
    def __init__(self, nc, args):
        self.args = args
        self.num_class = nc
        self.pred_list = []
        self.target_list = []

    def clear(self):
        self.pred_list = []
        self.target_list = []

    def __call__(self, pred, target):
        self.pred_list.append(pred)
        self.target_list.append(target)

    def get_class_acc(self):
        pred = torch.cat(self.pred_list, 0)
        target = torch.cat(self.target_list, 0)
        matrix = self.get_confusion_matrix(pred, target)
        correct = matrix.diagonal()
        return correct / matrix.sum(1)

    def get_named_class_acc(self):
        correct = self.get_class_acc()
        classNameMap = Datasets[self.args.dataset].Classes
        named_acc = {classNameMap[str(i)]: correct[i] for i in range(self.num_class)}
        return named_acc

    def get_confusion_matrix(self, prediction, target):
        """ Compute confusion matrix for prediction size=(nclass, nclass)
        :param prediction: (N, H, W) tensor
        :param target:     (N, H, W) tensor
        :param nclass:
        :return:
        """
        nclass = self.num_class
        prediction, target = to_numpy((prediction, target))
        assert prediction.shape == target.shape, \
            "Shape mismatch pred.shape={}, target.shape={}".format(prediction.shape, target.shape)
        mask = (target >= 0) & (target < nclass)
        label = nclass * target[mask].astype('int') + prediction[mask]
        count = np.bincount(label, minlength=nclass ** 2)
        confusion_matrix = count.reshape(nclass, nclass)
        return confusion_matrix


@LossFuncs.register('ce')
def CE_1D(logits, label):
    N = logits.size(0)
    logits = logits.view(N, -1)
    label = label.view(N)
    return F.cross_entropy(logits, label)


@LossFuncs.register('bce_onehot')
def BCE_onehot(logits, label, param=4):
    # print(num_classes)
    num_classes = int(param)
    N = logits.size(0)
    logits = logits.view(N, -1)
    label = label.view(N)
    label = F.one_hot(label, num_classes=num_classes).float()
    return F.binary_cross_entropy_with_logits(logits, label)


@LossFuncs.register('bce')
def BCE(logits, label):
    N = logits.size(0)
    logits = logits.view(-1)
    label = label.view(-1).float()
    return F.binary_cross_entropy_with_logits(logits, label)


@LossFuncs.register('balance_bce')
def balanced_BCE(logits, label):
    ratio = (label == 0).sum().float() / (label == 1).sum().float()
    logits = logits.view(-1)
    label = label.view(-1).float()
    return F.binary_cross_entropy_with_logits(logits, label, pos_weight=ratio)


@LossFuncs.register('l1')
def L1(logits, label):
    return (logits - label).abs().mean()


@LossFuncs.register('l2')
def MSE(logits, label):
    return ((logits - label) ** 2).mean()


@AccFuncs.register('acc')
def acc(logits, label):
    N = logits.size(0)
    logits = logits.view(N, -1)
    label = label.view(N).long()
    preds = logits.argmax(1)
    # print((preds == label))
    return (preds == label).sum().float() / N, N


@AccFuncs.register('bacc')
def b_acc(logits, label):
    logits = logits.view(-1)
    label = label.view(-1).long()
    N = logits.size(0)
    preds = (logits.sigmoid() > 0.5).long()
    # print((preds == label))
    return (preds == label).sum().float() / N, N


@AccFuncs.register('topk')
def topk(logits, label):
    # logits : N x c, label : N
    N = len(logits)
    k = 2
    # N x 2
    topk = torch.topk(logits, k, dim=-1)[1]
    label = label.view(N, 1)
    return (label == topk).sum().float() / N, N


@AccFuncs.register('c_topk')
def confidence_topk(logits, label, param=0.9):
    # logits : N x c, label : N
    threshold = int(param)
    pred = logits.argmax(1)
    conf = logits.softmax(1).max(1)[0]
    idx = conf >= threshold
    N = idx.sum()
    corrects = (pred[idx] == label[idx]).sum()

    N = len(logits)
    k = 2
    top2_pred = torch.topk(logits, k, dim=-1)[1][:, 1]  # N x 2
    corrects += (top2_pred[~idx] == label[~idx]).sum()
    # print(idx.sum(), len(logits))
    return corrects / N, N


@AccFuncs.register('rand_topk')
def randtopk(logits, label, param=2):
    N = len(logits)
    k = int(param)
    logits_topk, topk = torch.topk(logits, k, dim=-1)
    pred_idx = torch.randint(0, 2, (N,)).to(logits.devices)
    # prob_topk = logits_topk.softmax(1)
    # c = torch.distributions.Categorical(prob_topk)
    # pred_idx = c.sample()
    preds = topk.gather(1, pred_idx[:, None]).squeeze()
    return (preds == label).sum().float() / N, N


@LossFuncs.register('ce_lb')
def ce_lb(logits, label, smoothing=0.1, num_classes=7):
    # N x C
    gt = F.one_hot(label, num_classes).float()
    u = torch.ones_like(gt).float() / num_classes
    target_prob = (1 - smoothing) * gt + smoothing * u
    target_prob = target_prob.float().to(logits.devices)
    logprobs = F.log_softmax(logits, dim=-1)
    loss = - (target_prob * logprobs).sum(1).mean(0)
    return loss


@LossFuncs.register('kl')
def KL(pred, target):
    (mu1, sigma1), (mu2, sigma2) = pred, target
    a = (sigma1 / (sigma2 + 1e-5)).log() + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2 + 1e-5) - 0.5
    b = (sigma2 / (sigma1 + 1e-5)).log() + (sigma2 ** 2 + (mu2 - mu1) ** 2) / (2 * sigma1 ** 2 + 1e-5) - 0.5
    return a.mean(-1) + b.mean(-1)


def JS(feat1, feat2):
    feat1, feat2 = feat1.cpu().numpy(), feat2.cpu().numpy()
    return KL(feat1.mean(0), feat2.mean(0), feat1.std(0), feat2.std(0))


def info_nce_loss(features, batch_size, n_views, device, temperature):
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)
    print(labels)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


@LossFuncs.register('label_smooth')
def crossEntropyLabelSmooth(inputs, targets, param=0.1):
    epsilon = int(param)
    logsoftmax = nn.LogSoftmax(dim=1)
    num_classes = inputs.size()[1]
    log_probs = logsoftmax(inputs)
    targets = F.one_hot(targets, num_classes=num_classes).to(inputs.device).float()
    targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (- targets * log_probs).mean(0).sum()
    return loss


class JSD(nn.Module):

    def __init__(self):
        super(JSD, self).__init__()

    def forward(self, net_1_logits, net_2_logits):
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs = F.softmax(net_2_logits, dim=1)

        m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div(m.log(), net_1_probs, reduction="batchmean")
        loss += F.kl_div(m.log(), net_2_probs, reduction="batchmean")
        return (0.5 * loss)


"""
Analyze output utils
"""


def normalize_loss(d, out):
    if 'loss' in d:
        loss = d['loss']
        if 'size' not in d:
            if 'pred' in d:
                size = len(d['pred'])
            else:
                size = 1
        else:
            size = d['size']
    else:
        if 'loss_type' in d:
            assert 'pred' in d and 'target' in d
            loss = LossFuncs[d['loss_type']](d['pred'], d['target'])
            size = len(d['pred'])
        else:
            loss = None
            size = None

    out['loss_size'] = size
    out['loss'] = loss
    if 'weight' in d:
        out['loss'] = loss * d['weight']


def normalize_acc(d, out):
    if 'acc' in d:
        acc = d['acc']
        if 'size' not in d:
            if 'pred' not in d:
                size = 1
            else:
                size = len(d['pred'])
        else:
            size = d['size']
    else:
        if 'acc_type' in d:
            assert 'pred' in d and 'target' in d
            acc, size = AccFuncs[d['acc_type']](d['pred'], d['target'])
        else:
            acc = None
            size = None

    out['acc_size'] = size
    out['acc'] = acc


def analyze_output_dicts(output_dicts):
    output = {}
    for name, dic in output_dicts.items():
        if name in ['out', 'logits', 'log']:
            continue
        out = {}
        normalize_loss(dic, out)
        normalize_acc(dic, out)
        output[name] = out
    return output


def get_loss_and_acc(output_dicts, running_loss=None, running_acc=None, reduction='sum', prefix=None):
    outputs = analyze_output_dicts(output_dicts)
    total_loss = []
    for name, d in outputs.items():
        if prefix is not None:
            name = prefix + name
        loss, acc, acc_size, loss_size = d['loss'], d['acc'], d['acc_size'], d['loss_size']
        if loss is not None:
            total_loss.append(loss)
        if running_loss is not None and loss is not None:
            loss = to_numpy(loss)
            running_loss.update(name, loss.item() * loss_size, loss_size)
        if running_acc is not None and acc is not None:
            acc = to_numpy(acc)
            running_acc.update(name, acc * acc_size, acc_size)
    if len(total_loss) > 0:
        if reduction == 'sum':
            total_loss = torch.stack(total_loss).sum()
        elif reduction == 'mean':
            total_loss = torch.stack(total_loss).mean()
        elif reduction == 'none':
            pass
        else:
            raise Exception("Wrong reduction : {}".format(reduction))
        return total_loss
    else:
        return None
