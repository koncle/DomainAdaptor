import copy
import functools
import warnings

from framework.ERM import ERM
from framework.loss_and_acc import *
from framework.registry import EvalFuncs, Models
from models.AdaptorHeads import RotationHead, NormHead, NoneHead, Head, JigsawHead
from models.AdaptorHelper import get_new_optimizers, convert_to_target
from models.LAME import laplacian_optimization, kNN_affinity
from utils.tensor_utils import to, AverageMeterDict, zero_and_update


warnings.filterwarnings("ignore")
np.set_printoptions(edgeitems=30, linewidth=1000, formatter=dict(float=lambda x: "{:.4f},  ".format(x)))


class AdaMixBN(nn.BatchNorm2d):
    # AdaMixBn cannot be applied in an online manner.
    def __init__(self, in_ch, lambd=None, transform=True, mix=True, idx=0):
        super(AdaMixBN, self).__init__(in_ch)
        self.lambd = lambd
        self.rectified_params = None
        self.transform = transform
        self.layer_idx = idx
        self.mix = mix

    def get_retified_gamma_beta(self, lambd, src_mu, src_var, cur_mu, cur_var):
        C = src_mu.shape[1]
        new_gamma = (cur_var + self.eps).sqrt() / (lambd * src_var + (1 - lambd) * cur_var + self.eps).sqrt() * self.weight.view(1, C, 1, 1)
        new_beta = lambd * (cur_mu - src_mu) / (cur_var + self.eps).sqrt() * new_gamma + self.bias.view(1, C, 1, 1)
        return new_gamma.view(-1), new_beta.view(-1)

    def get_lambd(self, x, src_mu, src_var, cur_mu, cur_var):
        instance_mu = x.mean((2, 3), keepdims=True)
        instance_std = x.std((2, 3), keepdims=True)

        it_dist = ((instance_mu - cur_mu) ** 2).mean(1, keepdims=True) + ((instance_std - cur_var.sqrt()) ** 2).mean(1, keepdims=True)
        is_dist = ((instance_mu - src_mu) ** 2).mean(1, keepdims=True) + ((instance_std - src_var.sqrt()) ** 2).mean(1, keepdims=True)
        st_dist = ((cur_mu - src_mu) ** 2).mean(1)[None] + ((cur_var.sqrt() - src_var.sqrt()) ** 2).mean(1)[None]

        src_lambd = 1 - (st_dist) / (st_dist + is_dist + it_dist)

        src_lambd = torch.clip(src_lambd, min=0, max=1)
        return src_lambd

    def get_mu_var(self, x):
        C = x.shape[1]
        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)
        cur_mu = x.mean((0, 2, 3), keepdims=True)
        cur_var = x.var((0, 2, 3), keepdims=True)

        lambd = self.get_lambd(x, src_mu, src_var, cur_mu, cur_var).mean(0, keepdims=True)

        if self.lambd is not None:
            lambd = self.lambd

        if self.transform:
            if self.rectified_params is None:
                new_gamma, new_beta = self.get_retified_gamma_beta(lambd, src_mu, src_var, cur_mu, cur_var)
                # self.test(x, lambd, src_mu, src_var, cur_mu, cur_var, new_gamma, new_beta)
                self.weight.data = new_gamma.data
                self.bias.data = new_beta.data
                self.rectified_params = new_gamma, new_beta
            return cur_mu, cur_var
        else:
            new_mu = lambd * src_mu + (1 - lambd) * cur_mu
            new_var = lambd * src_var + (1 - lambd) * cur_var
            return new_mu, new_var

    def forward(self, x):
        n, C, H, W = x.shape
        new_mu = x.mean((0, 2, 3), keepdims=True)
        new_var = x.var((0, 2, 3), keepdims=True)

        if self.training:
            if self.mix:
                new_mu, new_var = self.get_mu_var(x)

            # Normalization with new statistics
            inv_std = 1 / (new_var + self.eps).sqrt()
            new_x = (x - new_mu) * (inv_std * self.weight.view(1, C, 1, 1)) + self.bias.view(1, C, 1, 1)
            return new_x
        else:
            return super(AdaMixBN, self).forward(x)

    def reset(self):
        self.rectified_params = None

    def test_equivalence(self, x):
        C = x.shape[1]
        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)
        cur_mu = x.mean((0, 2, 3), keepdims=True)
        cur_var = x.var((0, 2, 3), keepdims=True)
        lambd = 0.9

        new_gamma, new_beta = self.get_retified_gamma_beta(x, lambd, src_mu, src_var, cur_mu, cur_var)
        inv_std = 1 / (cur_var + self.eps).sqrt()
        x_1 = (x - cur_mu) * (inv_std * new_gamma.view(1, C, 1, 1)) + new_beta.view(1, C, 1, 1)

        new_mu = lambd * src_mu + (1 - lambd) * cur_mu
        new_var = lambd * src_var + (1 - lambd) * cur_var
        inv_std = 1 / (new_var + self.eps).sqrt()
        x_2 = (x - new_mu) * (inv_std * self.weight.view(1, C, 1, 1)) + self.bias.view(1, C, 1, 1)
        assert (x_2 - x_1).abs().mean() < 1e-5
        return x_1, x_2


class Losses():
    def __init__(self):
        self.losses = {
            'em': self.em,
            'slr': self.slr,
            'norm': self.norm,
            'gem-t': self.GEM_T,
            'gem-skd': self.GEM_SKD,
            'gem-aug': self.GEM_Aug,
        }

    def GEM_T(self, logits, **kwargs):
        logits = logits - logits.mean(1, keepdim=True).detach()
        T = logits.std(1, keepdim=True).detach() * 2
        prob = (logits / T).softmax(1)
        loss = - ((prob * prob.log()).sum(1) * (T ** 2)).mean()
        return loss

    def GEM_SKD(self, logits, **kwargs):
        logits = logits - logits.mean(1, keepdim=True).detach()
        T = logits.std(1, keepdim=True).detach() * 2

        original_prob = logits.softmax(1)
        prob = (logits / T).softmax(1)

        loss = - ((original_prob.detach() * prob.log()).sum(1) * (T ** 2)).mean()
        return loss

    def GEM_Aug(self, logits, **kwargs):
        logits = logits - logits.mean(1, keepdim=True).detach()
        T = logits.std(1, keepdim=True).detach() * 2
        aug_logits = kwargs['aug_logits']
        loss = - ((aug_logits.softmax(1).detach() * (logits / T).softmax(1).log()).sum(1) * (T ** 2)).mean()
        return loss

    def em(self, logits, **kwargs):
        prob = (logits).softmax(1)
        loss = (- prob * prob.log()).sum(1)
        return loss.mean()

    def slr(self, logits, **kwargs):
        prob = (logits).softmax(1)
        return -(prob * (1 / (1 - prob + 1e-8)).log()).sum(1).mean()  # * 3 is enough = 82.7

    def norm(self, logits, **kwargs):
        return -logits.norm(dim=1).mean() * 2

    def get_loss(self, name, **kwargs):
        return {name: {'loss': self.losses[name.lower()](**kwargs)}}


class EntropyMinimizationHead(Head):
    KEY = 'EM'
    ft_steps = 1

    def __init__(self, num_classes, in_ch, args):
        super(EntropyMinimizationHead, self).__init__(num_classes, in_ch, args)
        self.losses = Losses()

    def get_cos_logits(self, feats, backbone):
        w = backbone.fc.weight  # c X C
        w, feats = F.normalize(w, dim=1), F.normalize(feats, dim=1)
        logits = (feats @ w.t())  # / 0.07
        return logits

    def label_rectify(self, feats, logits, thresh=0.95):
        # mask = self.get_confident_mask(logits, thresh=thresh)
        max_prob = logits.softmax(1).max(1)[0]
        normed_feats = feats / feats.norm(dim=1, keepdim=True)
        # N x N
        sim = (normed_feats @ normed_feats.t()) / 0.07
        # sim = feats @ feats.t()
        # select from high confident masks
        selected_sim = sim  # * max_prob[None]
        # N x n @ n x C = N x C
        rectified_feats = (selected_sim.softmax(1) @ feats)
        return rectified_feats + feats

    def do_lame(self, feats, logits):
        prob = logits.softmax(1)
        unary = - torch.log(prob + 1e-10)  # [N, K]

        feats = F.normalize(feats, p=2, dim=-1)  # [N, d]
        kernel = kNN_affinity(5)(feats)  # [N, N]

        kernel = 1 / 2 * (kernel + kernel.t())

        # --- Perform optim ---
        Y = laplacian_optimization(unary, kernel)
        return Y

    def do_ft(self, backbone, x, label, loss_name=None, step=0, model=None, **kwargs):
        assert loss_name is not None

        if loss_name.lower() == 'gem-aug':
            with torch.no_grad():
                aug_x = kwargs['tta']
                n, N, C, H, W = aug_x.shape
                aug_x = aug_x.reshape(n * N, C, H, W)
                aug_logits = backbone(aug_x)[-1].view(n, N, -1).mean(1)
        else:
            aug_logits = None

        base_features = backbone(x)
        logits, feats = base_features[-1], base_features[-2].mean((2, 3))
        ret = {
            'main': {'acc_type': 'acc', 'pred': logits, 'target': label},
            'logits': logits.detach()
        }

        ret.update(self.losses.get_loss(loss_name, logits=logits, backbone=backbone, feats=feats,
                                        step=step, aug_logits=aug_logits))
        return ret

    def do_train(self, backbone, x, label, **kwargs):
        base_features = backbone(x)
        logits, feats = base_features[-1], base_features[-2].mean((2, 3))

        res = {
            'main': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': logits, 'target': label},
            'logits': logits.detach()
        }
        if self.args.LAME:
            res.update({'LAME': {'acc_type': 'acc', 'pred': self.do_lame(feats, logits), 'target': label}})
        return res

    def setup(self, model, online):
        model.backbone.train()
        lr = self.args.lr
        print(f'Learning rate : {lr}')
        return [
            get_new_optimizers(model, lr=lr, names=['bn'], opt_type='sgd', momentum=self.args.online),
        ]


@Models.register('DomainAdaptor')
class DomainAdaptor(ERM):
    def __init__(self, num_classes, pretrained=True, args=None):
        super(DomainAdaptor, self).__init__(num_classes, pretrained, args)
        heads = {
            'em': EntropyMinimizationHead,
            'rot': RotationHead,
            'norm': NormHead,
            'none': NoneHead,
            'jigsaw': JigsawHead,
        }
        self.head = heads[args.TTA_head.lower()](num_classes, self.in_ch, args)

        if args.AdaMixBN:
            self.bns = list(convert_to_target(self.backbone, functools.partial(AdaMixBN, transform=args.Transform, lambd=args.mix_lambda),
                                              verbose=False, start=0, end=5, res50=args.backbone == 'resnet50')[-1].values())

    def step(self, x, label, train_mode='test', **kwargs):
        if train_mode == 'train':
            res = self.head.do_train(self.backbone, x, label, model=self, **kwargs)
        elif train_mode == 'test':
            res = self.head.do_test(self.backbone, x, label, model=self, **kwargs)
        elif train_mode == 'ft':
            res = self.head.do_ft(self.backbone, x, label, model=self, **kwargs)
        else:
            raise Exception("Unexpected mode : {}".format(train_mode))
        return res

    def finetune(self, data, optimizers, loss_name, running_loss=None, running_corrects=None):
        if hasattr(self, 'bns'):
            [bn.reset() for bn in self.bns]

        with torch.enable_grad():
            res = None
            for i in range(self.head.ft_steps):
                o = self.step(**data, train_mode='ft', step=i, loss_name=loss_name)
                meta_train_loss = get_loss_and_acc(o, running_loss, running_corrects, prefix=f'A{i}_')
                zero_and_update(optimizers, meta_train_loss)
                if i == 0:
                    res = o
            return res

    def forward(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def setup(self, online):
        return self.head.setup(self, online)


@EvalFuncs.register('tta_ft')
def test_time_adaption(model, eval_data, lr, epoch, args, engine, mode):
    device, optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()

    model.eval()
    model_to_ft = copy.deepcopy(model)
    original_state_dict = model.state_dict()

    online = args.online
    optimizers = model_to_ft.setup(online)

    loss_names = args.loss_names  # 'gem-t', 'gem-skd', 'gem-tta']

    with torch.no_grad():
        for i, data in enumerate(eval_data):
            data = to(data, device)

            # Normal Test
            out = model(**data, train_mode='test')
            get_loss_and_acc(out, running_loss, running_corrects, prefix='original_')

            # test-time adaptation to a single batch
            for loss_name in loss_names:
                # recover to the original weight
                model_to_ft.load_state_dict(original_state_dict) if (not online) else ""

                # adapt to the current batch
                adapt_out = model_to_ft.finetune(data, optimizers, loss_name, running_loss, running_corrects)

                # get the adapted result
                cur_out = model_to_ft(**data, train_mode='test')

                get_loss_and_acc(cur_out, running_loss, running_corrects, prefix=f'{loss_name}_')
                if loss_name == loss_names[-1]:
                    get_loss_and_acc(cur_out, running_loss, running_corrects)  # the last one is recorded as the main result

    loss, acc = running_loss.get_average_dicts(), running_corrects.get_average_dicts()
    return acc['main'], (loss, acc)
