import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
from datasets import data_transforms
# from pointnet2_ops import pointnet2_utils
from torchvision import transforms
import utils.lr_sched as lr_sched


KNOWN_DATASETS = ['ScanobjNNcolor', 'ScanobjNNcolor_hardest', 'SWIModelNet']


tsne_visualize = False


train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


def intersection_and_union(output, target, K, ignore_index=-1):
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
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

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, val_dataloader), (_, test_dataloader), = \
                builder.dataset_builder_with_collate_fn(args, config.dataset.train), \
                builder.dataset_builder_with_collate_fn(args, config.dataset.val), \
                builder.dataset_builder_with_collate_fn(args, config.dataset.test)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_from_ckpts(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    if args.use_gpu:    
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
        model_without_ddp = base_model.module
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
        model_without_ddp = base_model.module
    # optimizer & scheduler
    optimizer, _ = builder.build_opti_sche_mae(model_without_ddp, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

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
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        global tsne_visualize
        if tsne_visualize:
            metrics = validate(base_model, val_dataloader, epoch, val_writer, args, config, logger=logger)
        for idx, data in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            lr_sched.adjust_learning_rate(optimizer, idx / n_batches + epoch, config)
            
            data_time.update(time.time() - batch_start_time)

            dataset_name = config.dataset.train._base_.NAME
            if dataset_name in KNOWN_DATASETS:
                for key, value in data[1].items():
                    data[1][key] = value.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            target = data['targets'].cuda().view(-1)
            # points = train_transforms(points)

            ret = base_model(data)

            # loss, acc = base_model.module.get_loss_acc(ret, label)
            loss, acc = base_model.module.get_loss_acc(ret, target)

            _loss = loss

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 10 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Acc = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        # if isinstance(scheduler, list):
        #     for item in scheduler:
        #         item.step(epoch)
        # else:
        #     scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_2', losses.avg(1), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0 and args.local_rank==0:
            # Validate the current model
            metrics = validate(base_model, val_dataloader, epoch, val_writer, args, config, logger=logger)

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)

            # Voting
            if args.voting and dataset_name in KNOWN_DATASETS:
                if (better and metrics.acc > args.voting_thres-2.0) or metrics.acc > args.voting_thres:
                    metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote, 'ckpt-best_vote', args, logger = logger)

        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
    if dataset_name in ['Scannet_seg']:
        print_log('[Final] Best mIoU = %.4f' % (best_metrics.acc), logger=logger)
        print_log('[Final] Best vote mIoU = %.4f' % (best_metrics_vote.acc), logger=logger)
    else:
        print_log('[Final] Best acc = %.4f' % (best_metrics.acc), logger=logger)
        print_log('[Final] Best vote acc = %.4f' % (best_metrics_vote.acc), logger=logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, val_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    global tsne_visualize
    if tsne_visualize:
        test_logits = []
    with torch.no_grad():
        for idx, data in enumerate(val_dataloader):
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name in KNOWN_DATASETS:
                for key, value in data[1].items():
                    data[1][key] = value.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            target = data['targets'].cuda().view(-1)

            logits = base_model(data)

            if tsne_visualize:
                test_logits.append(logits.detach())

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target)

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)
        if tsne_visualize:
            test_logits = torch.cat(test_logits, dim=0)

        # TODO: support distributed evaluation over multiple hosts
        # if args.distributed:
        #     test_pred = dist_utils.gather_tensor(test_pred, args)
        #     test_label = dist_utils.gather_tensor(test_label, args)
        #     if tsne_visualize:
        #         test_logits = dist_utils.gather_tensor(test_logits, args)

    # Add testing results to TensorBoard
    if dataset_name in ['Scannet_seg']:
        intersection, union, target = intersection_and_union(
                test_pred.detach().cpu().numpy(), test_label.detach().cpu().numpy(), config.num_classes, -1
            )
        mask = union != 0
        iou_class = intersection / (union + 1e-10)
        miou = np.mean(iou_class[mask]) * 100
        print_log(f'[Validation] EPOCH: {epoch}  IoU class: {iou_class}', logger=logger)
        print_log('[Validation] EPOCH: %d  mIoU = %.4f' % (epoch, miou), logger=logger)
        if val_writer is not None:
            val_writer.add_scalar('Metric/mIoU', miou, epoch)
        ret_metric = Acc_Metric(miou)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)
        if val_writer is not None:
            val_writer.add_scalar('Metric/ACC', acc, epoch)
    else:
        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)
        if val_writer is not None:
            val_writer.add_scalar('Metric/ACC', acc, epoch)
        ret_metric = Acc_Metric(acc)

        if tsne_visualize:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            def plot_embedding(data, label):
                x_min, x_max = np.min(data, 0), np.max(data, 0)
                data = (data - x_min) / (x_max - x_min)

                fig = plt.figure()
                ax = plt.subplot(111)
                plt.scatter(data[:, 0], data[:, 1], c=label, cmap='Spectral')
                # for i in range(data.shape[0]):
                #     plt.scatter(data[i, 0], data[i, 1], c=plt.cm.Set1(label[i] / 16.))
                    # plt.text(data[i, 0], data[i, 1], str(label[i]),
                    #         color=plt.cm.Set1(label[i] / 16.),
                    #         fontdict={'weight': 'bold', 'size': 9})
                plt.xticks([])
                plt.yticks([])
                return fig
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            result = tsne.fit_transform(test_logits.cpu().numpy())
            fig = plot_embedding(result, test_label.cpu().numpy())
            # plt.show(fig)
            plt.savefig('test')
            exit(0)

    # if args.distributed:
    #     torch.cuda.synchronize()

    return ret_metric


def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    with torch.no_grad():
        for kk in range(times):
            epoch_pred = []
            for idx, data in enumerate(test_dataloader):
                dataset_name = config.dataset.val._base_.NAME
                if dataset_name in KNOWN_DATASETS:
                    for key, value in data[1].items():
                        data[1][key] = value.cuda()
                else:
                    raise NotImplementedError(f'Train phase do not support {dataset_name}')

                logits = base_model(data)
                epoch_pred.append(logits.detach())
                if kk==times-1:
                    test_label.append(data['targets'].cuda().view(-1))
            
            test_pred.append(torch.cat(epoch_pred, dim=0).unsqueeze(0))

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)
        test_pred = test_pred.mean(0)
        _, test_pred = torch.max(test_pred, dim=-1)

        # if args.distributed:
        #     test_pred = dist_utils.gather_tensor(test_pred, args)
        #     test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        # if args.distributed:
        #     torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)

    return Acc_Metric(acc)