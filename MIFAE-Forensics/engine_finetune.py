# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc

import joblib


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, criterion_recon=None, trained_mae=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if args.recon_loss:
        metric_logger.add_meter('recon_loss', misc.SmoothedValue())
        metric_logger.add_meter('ce_loss', misc.SmoothedValue())


    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # for data_iter_step, (samples, targets) in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        labels = targets

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

            if args.recon_loss or args.recon_real_loss:
                if type(model).__name__ == 'DistributedDataParallel':
                    model = model.module
                if type(trained_mae).__name__ == 'DistributedDataParallel':
                    trained_mae = trained_mae.module

                output_features = model.forward_all_features(samples)

                # applying decoder

                # embed tokens
                x = trained_mae.decoder_embed(output_features)

                # add pos embed
                x = x + trained_mae.decoder_pos_embed

                # apply Transformer blocks
                for blk in trained_mae.decoder_blocks:
                    x = blk(x)
                x = trained_mae.decoder_norm(x)

                # predictor projection
                x = trained_mae.decoder_pred(x)

                # remove cls token
                x = x[:, 1:, :]
                target = trained_mae.patchify(samples)

                # Compute reconstruction loss
                recon_loss = criterion_recon(x, target, labels)

                ce_loss_value = loss.item()
                recon_loss_value = recon_loss.item()

                loss = loss + args.recon_weight * recon_loss.mean()

        loss_value = loss.item()

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if args.recon_loss:
            metric_logger.update(recon_loss=recon_loss_value)
            metric_logger.update(ce_loss=ce_loss_value)
            ml_loss_reduce = misc.all_reduce_mean(recon_loss_value)
            ce_loss_reduce = misc.all_reduce_mean(ce_loss_value)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)

            if args.recon_loss:
                log_writer.add_scalar('recon_loss', ml_loss_reduce, epoch_1000x)
                log_writer.add_scalar('ce_loss', ce_loss_reduce, epoch_1000x)
                log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            else:
                log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)

            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(dataset_val, data_loader, model, epoch, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    val_loss = 0
    val_corrects = 0
    # switch to evaluation mode
    model.eval()

    # for i, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
    for i, batch in enumerate(tqdm(data_loader)):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

            val_loss += loss.data.item()
            _, y_preds = torch.max(output.data, dim=1)
            val_corrects += torch.sum(y_preds == target.data).to(torch.float32)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

        if i == 0:
            total_outputs = np.array(output.cpu())
            total_labels = np.array(target.cpu())

        else:
            total_outputs = np.concatenate((total_outputs, output.cpu()), axis=0)
            total_labels = np.concatenate((total_labels, target.cpu()), axis=0)

    fpr, tpr, threshold = roc_curve(total_labels, total_outputs[:, 1], pos_label=1)
    epoch_auc = auc(fpr, tpr)
    epoch_eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    epoch_loss = val_loss / len(dataset_val)
    epoch_acc = val_corrects.item() / len(dataset_val)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print('epoch: {} epoch val loss: {:.4f} Acc: {:.4f} AUC: {:.4f} EER: {:.4f}'.format(epoch, epoch_loss, epoch_acc,
                                                                                        epoch_auc, epoch_eer))
  
    return {'loss': epoch_loss, 'acc': epoch_acc, 'auc': epoch_auc, 'eer': epoch_eer}
