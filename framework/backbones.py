import torch.nn as nn
from torchvision.models import AlexNet
from torchvision.models import resnet18, resnet50, alexnet
import torch.nn.functional as F

__all__ = ['AlexNet', 'Resnet']

from framework.registry import Backbones
from models.DomainAdaptor import AdaMixBN


def init_classifier(fc):
    nn.init.xavier_uniform_(fc.weight, .1)
    nn.init.constant_(fc.bias, 0.)
    return fc


@Backbones.register('resnet50')
@Backbones.register('resnet18')
class Resnet(nn.Module):
    def __init__(self, num_classes, pretrained=False, args=None):
        super(Resnet, self).__init__()
        if '50' in args.backbone:
            print('Using resnet-50')
            resnet = resnet50(pretrained=pretrained)
            self.in_ch = 2048
        else:
            resnet = resnet18(pretrained=pretrained)
            self.in_ch = 512
        self.conv1 = resnet.conv1
        self.relu = resnet.relu
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(self.in_ch, num_classes, bias=False)
        if args.in_ch != 3:
            self.init_conv1(args.in_ch, pretrained)

    def init_conv1(self, in_ch, pretrained):
        model_inplanes = 64
        conv1 = nn.Conv2d(in_ch, model_inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        old_weights = self.conv1.weight.data
        if pretrained:
            for i in range(in_ch):
                self.conv1.weight.data[:, i, :, :] = old_weights[:, i % 3, :, :]
        self.conv1 = conv1

    def forward(self, x):
        net = self
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)

        l1 = net.layer1(x)
        l2 = net.layer2(l1)
        l3 = net.layer3(l2)
        l4 = net.layer4(l3)
        logits = self.fc(l4.mean((2, 3)))
        return x, l1, l2, l3, l4, logits

    def get_lr(self, fc_weight):
        lrs = [
            ([self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4], 1.0),
            (self.fc, fc_weight)
        ]
        return lrs


@Backbones.register('alexnet')
class Alexnet(nn.Module):
    # PACS : (88.08+60.74+63.44+54.31)/4
    # VLCS :       (95.12+59.75+65.46+65.45)/4=71.45
    # VLCS(1e-4) : (95.97+56.55+67.54+63.80)/4=70.96
    def __init__(self, num_classes, pretrained=True, args=None):
        super(Alexnet, self).__init__()
        self.args = args
        cur_alexnet = alexnet(pretrained=pretrained)
        self.features = cur_alexnet.features
        self.avgpool = cur_alexnet.avgpool
        self.feature_layers = nn.Sequential(*list(cur_alexnet.classifier.children())[:-1])
        self.in_ch = cur_alexnet.classifier[-1].in_features
        self.fc = nn.Linear(self.in_ch, num_classes, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        feats = self.feature_layers(x)
        output_class = self.fc(feats)
        return feats, output_class

    def get_lr(self, fc_weight):
        return [([self.features, self.feature_layers], 1.0), (self.fc, fc_weight)]


class Convolution(nn.Module):

    def __init__(self, c_in, c_out, mixbn=False):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        if mixbn:
            self.bn = AdaMixBN(c_out)
        else:
            self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.seq = nn.Sequential(
            self.conv,
            self.bn,
            self.relu
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@Backbones.register('convnet')
class ConvNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, args=None):
        super(ConvNet, self).__init__()

        c_hidden = 64
        mix = True
        self.conv1 = Convolution(3, c_hidden, mixbn=mix)
        self.conv2 = Convolution(c_hidden, c_hidden, mixbn=mix)
        self.conv3 = Convolution(c_hidden, c_hidden, mixbn=mix)
        self.conv4 = Convolution(c_hidden, c_hidden, mixbn=mix)

        self._out_features = 2**2 * c_hidden
        self.in_ch = 3
        self.fc = nn.Linear(self._out_features, num_classes)

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert (H == 32 and W == 32), "Input to network must be 32x32, " "but got {}x{}".format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        feat = x
        x = x.view(x.size(0), -1)
        return x[:, :, None, None], self.fc(x)

    def get_lr(self, fc_weight):
        return [(self, 1.0)]
