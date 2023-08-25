import torch
import torch.nn as nn

from framework.registry import Models, Datasets, Backbones

__all__ = ['ERM']


@Models.register('ERM')
class ERM(nn.Module):
    def __init__(self, num_classes, pretrained, args):
        super(ERM, self).__init__()
        self.args = args
        self.backbone = Backbones[args.backbone](num_classes, pretrained, args)
        self.num_classes = num_classes
        self.in_ch = self.backbone.in_ch

    def load_pretrained(self, path=None, prefix=None, absolute=False):
        if path is None:
            path = self.args.path
        if not absolute:
            cur_domain = Datasets[self.args.dataset].Domains[self.args.exp_num[0]]
            path = str(path) + '/{}'.format(cur_domain) + str(self.args.time) + '/models/model_best.pt'
        state = torch.load(path, map_location='cpu')
        if 'model' in state.keys():
            state = state['model']
        elif 'state_dict' in state.keys():
            state = state['state_dict']
            state['backbone.fc.weight'] = state['classifier.weight']
        elif 'encoder_state_dict' in state.keys():
            backbone = {'backbone.'+k : v for k,v  in state['encoder_state_dict'].items()}
            backbone['backbone.fc.weight'] = state['classifier_state_dict']['layers.weight']
            state = backbone

        keys = list(state.keys())
        if 'module' in keys[0]:
            state = {k.replace('module', 'backbone'): state[k] for k in keys}

        if 'resnet' in keys[len(keys)//2]:
            state = {k.replace('resnet', 'backbone'): state[k] for k in keys}

        if prefix is not None:
            new_state = {}
            for k, v in state.items():
                new_state.update({prefix + k: v})
            state = new_state
        ret = self.load_state_dict(state, strict=False)
        print('load from {}, state : {}'.format(path, ret))
        return state

    def forward(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def get_lr(self, fc_weight):
        old_lr = self.backbone.get_lr(fc_weight)

        new_params = []
        for name, child in self.named_children():
            if 'backbone' not in name:
                if hasattr(child, 'get_lr'):
                    old_lr.extend(child.get_lr(fc_weight))
                else:
                    new_params.append([child, fc_weight])
        old_lr.extend(new_params)
        return old_lr

    def step(self, x, label, **kwargs):
        l4, final_logits = self.backbone(x)[-2:]
        return {
            'main': {'loss_type': 'ce', 'acc_type': 'acc', 'pred': final_logits, 'target': label, },
            'logits': final_logits,
            'feat' : [l4.mean((2,3))]
        }
