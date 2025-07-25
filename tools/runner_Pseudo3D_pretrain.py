import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import math
from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from torch.cuda.amp import autocast, GradScaler


class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    if not args.eval_linearsvm:
        (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder_with_collate_fn(args, config.dataset.train), \
                                                                builder.dataset_builder_with_collate_fn(args, config.dataset.val)
    else:
        (_, test_dataloader) = builder.dataset_builder_with_collate_fn(args, config.dataset.val)
    (_, extra_train_dataloader)  = builder.dataset_builder_with_collate_fn(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        checkpoint = torch.load(args.start_ckpts, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.start_ckpts)
        if 'model' in checkpoint.keys():
            checkpoint_model = checkpoint['model']
        elif 'base_model' in checkpoint.keys():
            checkpoint_model = checkpoint['base_model']
        state_dict = base_model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = base_model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=False)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()

    if args.eval_linearsvm:
        if args.start_ckpts is not None:
            if args.local_rank==0:
                metrics = validate(base_model, extra_train_dataloader, test_dataloader, 0, val_writer, args, config, logger=logger)
            return
        else:
            raise RuntimeError("eval_linearsvm mode but start_ckpts is not given!")

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss1'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, data in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name in ['Pseudo3D', 'Pseudo3DCC3M', 'SWIShapeNet']:
                for key, value in data[1].items():
                    data[1][key] = value.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            
            with torch.cuda.amp.autocast(enabled=args.amp):
                _loss = base_model(data)

            if args.amp:
                scaler.scale(_loss).backward()
                if num_iter == config.step_per_update:
                    num_iter = 0
                    scaler.step(optimizer)
                    scaler.update()
                    base_model.zero_grad()
            else:
                _loss.backward()
                if num_iter == config.step_per_update:
                    num_iter = 0
                    optimizer.step()
                    base_model.zero_grad()

            if args.distributed:
                _loss = dist_utils.reduce_tensor(_loss, args)
                losses.update([_loss.item()])
            else:
                losses.update([_loss.item()])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss_1', _loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0 and args.local_rank==0:
            # Validate the current model
            metrics = validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    with torch.no_grad():
        for idx, data in enumerate(extra_train_dataloader):
            dataset_name = config.dataset.extra_train._base_.NAME
            if dataset_name in ['ScanobjNNcolor', 'SWIModelNet']:
                for key, value in data[1].items():
                    data[1][key] = value.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            batch_size = data['batch_size']
            feature = base_model(data, inference=True).reshape(batch_size, -1)
            target = data['targets'].cuda().view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, data in enumerate(test_dataloader):
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name in ['ScanobjNNcolor', 'SWIModelNet']:
                for key, value in data[1].items():
                    data[1][key] = value.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            batch_size = data['batch_size']
            feature = base_model(data, inference=True).reshape(batch_size, -1)
            target = data['targets'].cuda().view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())


        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        # if args.distributed:
        #     train_features = dist_utils.gather_tensor(train_features, args)
        #     train_label = dist_utils.gather_tensor(train_label, args)
        #     test_features = dist_utils.gather_tensor(test_features, args)
        #     test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, svm_acc), logger=logger)

        # if args.distributed:
        #     torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)