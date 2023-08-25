from .registry import Schedulers
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

Schedulers.register('step', StepLR)
Schedulers.register('exp', ExponentialLR)
Schedulers.register('cos', CosineAnnealingLR)

# import files to access them from registry
from framework.backbones import Resnet
from framework import basic_train_funcs, ERM, basic_train_funcs, loss_and_acc
import dataloader
import models
# dataloader will be imported from other files

