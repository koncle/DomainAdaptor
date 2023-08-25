import ast
import sys
import argparse

from framework.exp import Experiments


def get_default_parser():
    dataset = 'PACS'
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-entry', action='store_true')

    parser.add_argument('--data-root', default='/data/DataSets/')
    parser.add_argument('--dataset', default='{}'.format(dataset))
    parser.add_argument('--save-path', default='../script/{}_New/resnet_test'.format(dataset))
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--model', default='ERM')
    parser.add_argument('--train', default='deepall')
    parser.add_argument('--eval', default='deepall')

    parser.add_argument('--exp-num', nargs='+', type=int, default=[1],
                        help='num >= 0 select which domain to train, num == -1 to train all domains,  num == -2 to trian all domains multi times. ')
    parser.add_argument('--start-time', type=int, default=0)
    parser.add_argument('--times', type=int, default=1)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-epoch', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)

    # almost no need to change
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--save-step', type=int, default=1000)  # Save steps
    parser.add_argument('--start-save-epoch', type=int, default=1000)  # Save steps
    parser.add_argument('--save-last', action='store_true')

    # scheduler
    parser.add_argument('--scheduler', default='step')
    parser.add_argument('--lr-decay-gamma', type=float, default=0.1)

    # optimizer
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--fc-weight', type=float, default=10.0)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--opt-split', action='store_true')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--nesterov', type=ast.literal_eval, default=True)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    parser.add_argument('--in-ch', default=3, type=int)

    # dataset
    parser.add_argument('--loader', default='normal', choices=['normal', 'meta', 'original', 'interleaved'])
    parser.add_argument('--img-size', default=224, type=int)
    parser.add_argument('--color-jitter', type=ast.literal_eval, default=True)  # important
    parser.add_argument('--min-scale', type=float, default=0.8)
    parser.add_argument('--domain-label', action='store_true')
    parser.add_argument('--data-path', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--src', nargs='+', type=int, default=[-1])
    parser.add_argument('--tgt', nargs='+', type=int, default=[-1])
    parser.add_argument('--do-train', type=ast.literal_eval, default=True)
    parser.add_argument('--do-not-transform', action='store_true')
    parser.add_argument('--load-path', type=str, default='')
    parser.add_argument('--shuffled', type=ast.literal_eval, default=True)
    parser.add_argument('--test-with-eval', action='store_true')
    parser.add_argument('--small-dataset', action='store_true')

    # ------ customized parameters ------
    parser.add_argument('--TN', action='store_true')
    parser.add_argument('--meta-step', default=1, type=int)
    parser.add_argument('--meta-lr', default=1e-3, type=float)
    parser.add_argument('--meta-lr-weight', default=1, type=float)
    parser.add_argument('--meta-second-order', type=ast.literal_eval, default=False)
    parser.add_argument('--meta-aug', default=1, type=float)

    parser.add_argument('--replace', action='store_true')

    parser.add_argument('--TTAug', action='store_true')
    parser.add_argument('--TTA-bs', default=3, type=int)
    parser.add_argument('--TTA-head', default='em')

    # augment data in dataset
    parser.add_argument('--jigsaw', action='store_true')
    parser.add_argument('--rot', action='store_true')

    # loss list
    parser.add_argument('--head', type=str, default='em', help='Classification for DomainAdaptor')
    parser.add_argument('--loss-names', nargs='+', type=str, default=['gem-t'])

    # AdaMixBN
    parser.add_argument('--AdaMixBN', action='store_true', default=True)
    parser.add_argument('--Transform', action='store_true', default=True)
    parser.add_argument('--mix-lambda', type=float, default=None)

    parser.add_argument('--LAME', action='store_true', default=False)
    parser.add_argument('--online', action='store_true', default=False)

    return parser


if __name__ == '__main__':
    parser = get_default_parser()
    args = parser.parse_args()
    exp = Experiments(args)
    exp.run()
