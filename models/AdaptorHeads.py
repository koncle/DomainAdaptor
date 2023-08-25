import torch
from torch import nn

from models.AdaptorHelper import get_new_optimizers


class Head(nn.Module):
    Replace = False
    ft_steps = 1

    def __init__(self, num_classes, in_ch, args):
        super(Head, self).__init__()
        self.args = args
        self.in_ch = in_ch
        self.num_classes = num_classes

    def forward(self, base_features, x, label, backbone, **kwargs):
        raise NotImplementedError()

    def setup(self, whole_model, online):
        whole_model.backbone.train()
        lr = 0.05
        print(f'Learning rate : {lr}')
        return [
            get_new_optimizers(whole_model, lr=lr, names=['bn'], opt_type='sgd')
        ]

    def do_ft(self, backbone, x, label, **kwargs):
        return self.do_train(backbone, x, label, **kwargs)

    def do_test(self, backbone, x, label, **kwargs):
        return self.do_train(backbone, x, label, **kwargs)

    def do_train(self, backbone, x, label, **kwargs):
        base_features = backbone(x)
        class_dict = {'main': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': base_features[-1], 'target': label}}
        return class_dict


class RotationHead(Head):
    KEY = 'rotation'

    def setup(self, whole_model, online):
        whole_model.backbone.train()
        lr = 0.05
        print(f'Learning rate : {lr}')
        return [
            get_new_optimizers(whole_model, lr=lr, names=['bn'], opt_type='sgd')
        ]

    def __init__(self, num_classes, in_ch, args):
        super(RotationHead, self).__init__(num_classes, in_ch, args)
        self.shared = args.shared
        self.rotation_fc = nn.Linear(512, 4, bias=False)
        emb_dim = in_ch
        # self.rotation_fc = nn.Sequential(
        #     nn.Linear(in_ch, emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(emb_dim, emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(emb_dim, 4),
        # )

    def do_ft(self, backbone, x, label, **kwargs):
        logits = backbone(x)[-1]

        rotated_x, rotation_label = kwargs['rot_x'], kwargs['rot_label']
        l4 = backbone(rotated_x)[-2].mean((-1, -2))
        rotation_logits = self.rotation_fc(l4)

        return {
            'main': {'acc_type': 'acc', 'pred': logits, 'target': label},
            'rotation': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': rotation_logits, 'target': rotation_label}
        }

    def do_train(self, backbone, x, label, **kwargs):
        base_features = backbone(x)

        rotated_x, rotation_label = kwargs['rot_x'], kwargs['rot_label']
        l4 = backbone(rotated_x)[-2].mean((-1, -2))

        rotation_logits = self.rotation_fc(l4)

        class_dict = {
            'main': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': base_features[-1], 'target': label},
            'rotation': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': rotation_logits, 'target': rotation_label, 'weight':0.0}
        }
        return class_dict


class NormHead(Head):
    KEY = 'Norm'

    def __init__(self, num_classes, in_ch, args):
        super(NormHead, self).__init__(num_classes, in_ch, args)

        class MLP(nn.Module):
            def __init__(self, in_size=10, out_size=1, hidden_dim=32, norm_reduce=False):
                super(MLP, self).__init__()
                self.norm_reduce = norm_reduce
                self.model = nn.Sequential(
                    nn.Linear(in_size, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_size),
                )

            def forward(self, x):
                out = self.model(x)
                if self.norm_reduce:
                    out = torch.norm(out)
                return out

        self.mlp = MLP(in_size=num_classes, norm_reduce=True)

    def do_ft(self, backbone, x, label, **kwargs):
        base_features = backbone(x)
        feats = base_features[-1]
        normed_loss = self.mlp(feats)
        return {
            'main': {'acc_type': 'acc', 'pred': base_features[-1], 'target': label},
            'norm_loss': {'loss': normed_loss},
        }

    def do_train(self, backbone, x, label, **kwargs):
        base_features = backbone(x)
        feats = base_features[-1]
        normed_loss = self.mlp(feats)
        return {
            'main': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': base_features[-1], 'target': label},
            NormHead.KEY: {'loss': normed_loss}
        }


class JigsawHead(Head):
    KEY = 'Jigsaw'

    def __init__(self, num_classes, in_ch, args):
        super(JigsawHead, self).__init__(num_classes, in_ch, args)
        jigsaw_classes = 32
        emb_dim = in_ch
        # self.jigsaw_classifier = nn.Linear(in_ch, jigsaw_classes)
        self.jigsaw_classifier = nn.Sequential(
            nn.Linear(in_ch, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, jigsaw_classes),
        )
        self.i = 0

    def do_ft(self, backbone, x, label, **kwargs):
        base_features = backbone(x)
        logits = base_features[-1]

        jig_features = backbone(kwargs['jigsaw_x'])[-2]
        jig_features = jig_features.mean((-1, -2))
        jig_logits = self.jigsaw_classifier(jig_features)
        return {
            # 'main': {'acc_type': 'acc', 'pred': logits, 'target': label},
            'jig': {'acc_type': 'acc', 'pred': jig_logits, 'target': kwargs['jigsaw_label'], 'loss_type': 'ce'},
        }

    def train(self, mode=True):
        super(JigsawHead, self).train(mode)
        self.i = 0

    def do_train(self, backbone, x, label, **kwargs):
        base_features = backbone(x)
        logits = base_features[-1]
        ret = {
                'main': {'acc_type': 'acc', 'pred': logits, 'target': label, 'loss_type': 'ce'},
            }
        # if self.i == 0 or random.random() > 0.9:
        #     self.i = 1
        if True:
            jig_features = backbone(kwargs['jigsaw_x'])
            jig_class_logits = jig_features[-1]
            jig_features = jig_features[-2].mean((-1, -2))
            jig_logits = self.jigsaw_classifier(jig_features)
            ret.update({
                    'jig': {'acc_type': 'acc', 'pred': jig_logits, 'target': kwargs['jigsaw_label'], 'loss_type': 'ce', 'weight':0.1},
                    # 'jig_cls': {'acc_type': 'acc', 'pred': jig_class_logits, 'target': label, 'loss_type': 'ce', 'weight':0.5},
                })
        return ret

    def setup(self, whole_model, online):
        whole_model.backbone.train()
        # online best : 0.01
        # not online  : 0.02?
        lr = 0.01 # 0.0005 is better for MDN
        print(f"Learning rate : {lr} ")
        return get_new_optimizers(whole_model, lr=lr, names=['bn'], opt_type='sgd', momentum=online)


class NoneHead(Head):
    def do_train(self, backbone, x, label, **kwargs):
        base_features = backbone(x)
        return {
            'main': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': base_features[-1], 'target': label},
        }

