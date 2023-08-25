from framework.loss_and_acc import *
from framework.meta_util import split_image_and_label
from framework.registry import EvalFuncs, TrainFuncs
from models.AdaptorHelper import get_new_optimizers
from utils.tensor_utils import to, AverageMeterDict

"""
ARM
"""
@TrainFuncs.register('tta_meta')
def tta_meta_train2(meta_model, train_data, lr, epoch, args, engine, mode):
    import higher
    device, optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()

    inner_opt_conv = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], momentum=False)
    meta_model.train()
    print(f'Meta LR : {args.meta_lr}')

    for data_list in train_data:
        data_list = to(data_list, device)
        split_data = split_image_and_label(data_list, size=args.batch_size)

        for data in split_data:

            with higher.innerloop_ctx(meta_model, inner_opt_conv, copy_initial_weights=False, track_higher_grads=True) as (fnet, diffopt):
                for _ in range(args.meta_step):
                    unsup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'adapt{_}_')
                    diffopt.step(unsup_loss)
                main_loss, unsup_loss = get_loss_and_acc(fnet(**data, train_mode='train'), running_loss, running_corrects, reduction='none')
                (main_loss).backward()
            optimizers.step()
            optimizers.zero_grad()

    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()


@EvalFuncs.register('tta_meta')
def tta_meta_test(meta_model, eval_data, lr, epoch, args, engine, mode):
    import higher
    device, optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()

    inner_opt = get_new_optimizers(meta_model, lr=args.meta_lr, names=['bn'], momentum=False)
    meta_model.eval()
    for data in eval_data:
        data = to(data, device)

        with torch.no_grad():  # Normal Test
            get_loss_and_acc(meta_model.step(**data, train_mode='test'), running_loss, running_corrects, prefix='original_')

        with higher.innerloop_ctx(meta_model, inner_opt, copy_initial_weights=False, track_higher_grads=False) as (fnet, diffopt):
            fnet.train()
            for _ in range(args.meta_step):
                unsup_loss = get_loss_and_acc(fnet(**data, train_mode='ft', step=_), running_loss, running_corrects, prefix=f'adapt{_}_')
                diffopt.step(unsup_loss)
            get_loss_and_acc(fnet(**data, train_mode='test'), running_loss, running_corrects)

    loss, acc = running_loss.get_average_dicts(), running_corrects.get_average_dicts()
    if 'main' in acc:
        return acc['main'], (loss, acc)
    else:
        return 0, (loss, acc)
