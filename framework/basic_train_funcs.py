import torch

from framework.loss_and_acc import get_loss_and_acc
from framework.meta_util import meta_learning_MAML, split_image_and_label
from framework.registry import EvalFuncs, TrainFuncs
from utils.tensor_utils import to, AverageMeterDict, zero_and_update


@TrainFuncs.register('meta')
@TrainFuncs.register('deepall')
def deepall_train(model, train_data, lr, epoch, args, engine, mode):
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    optimizers, device = engine.optimizers, engine.device

    model.train()
    for i, data_list in enumerate(train_data):
        data_list = to(data_list, device)
        output_dicts = model(**data_list, epoch=epoch, step=len(train_data) * epoch + i, engine=engine, train_mode='train')
        total_loss = get_loss_and_acc(output_dicts, running_loss, running_corrects)
        if total_loss is not None:
            total_loss.backward()

        optimizers.step()
        optimizers.zero_grad()
    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()


@EvalFuncs.register('meta')
@EvalFuncs.register('deepall')
def deepall_eval(model, eval_data, lr, epoch, args, engine, mode):
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    device = engine.device

    if hasattr(model, 'swa'):
        print('Eval with swa : ', end='')
        model = model.swa.module

    model.train() if args.TN else model.eval()

    with torch.no_grad():
        for i, data_list in enumerate(eval_data):
            data_list = to(data_list, device)
            outputs = model(**data_list, epoch=epoch, step=len(eval_data) * epoch + i, engine=engine, train_mode='test')
            get_loss_and_acc(outputs, running_loss, running_corrects)
    loss, acc = running_loss.get_average_dicts(), running_corrects.get_average_dicts()
    if 'main' in acc:
        return acc['main'], (loss, acc)
    else:
        return 0, (loss, acc)


@TrainFuncs.register('mldg')
def mldg(meta_model, train_data, meta_lr, epoch, args, engine, mode):
    assert args.loader == 'meta'
    device, optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    print('Meta lr : {}, loops : {}'.format(meta_lr, len(train_data)))

    meta_model.train()
    for data_list in train_data:
        meta_train_data, meta_test_data = split_image_and_label(to(data_list, device), size=args.batch_size, loo=True)

        with meta_learning_MAML(meta_model) as fast_model:
            for j in range(args.meta_step):
                meta_train_loss = get_loss_and_acc(fast_model.step(**meta_train_data), running_loss, running_corrects)
                fast_model.meta_step(meta_train_loss, meta_lr, use_second_order=args.meta_second_order)

            meta_val_loss = get_loss_and_acc(fast_model.step(**meta_test_data), running_loss, running_corrects)

        zero_and_update(optimizers, (meta_train_loss+meta_val_loss))
    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()

