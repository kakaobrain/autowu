"""Modified from Pytorch Image Models (https://github.com/rwightman/pytorch-image-models/)"""
import argparse
from datetime import datetime
import logging
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from autowu import AutoWU
from timm.data import create_transform, resolve_data_config
from timm.loss import LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name
from timm.optim import AdamP
import timm.utils
from torchvision.datasets import ImageNet
import utils

log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stderr)


class MODELS:
    RESNET = 'resnet50'
    EFFNET = 'efficientnet_b0'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default=MODELS.RESNET,
                        choices=[MODELS.RESNET, MODELS.EFFNET])
    parser.add_argument('--decay-sched', type=str, default='cos',
                        choices=['cos', 'const-cos'])
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size per gpu')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='weight decay (for AdamP)')
    parser.add_argument('--epochs', type=int, default=120)

    parser.add_argument('--data-root', type=str, default='./data/',
                        help='imagenet dataset root')
    parser.add_argument('--workers', type=int, default=8)

    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-dir', type=str, default='./result/')
    parser.add_argument('--save-freq', type=int, default=20)

    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Used for multi-process training. Can either be manually set '
                             'or automatically set by using \'python -m torch.distributed.launch\'.')
    args = parser.parse_args()

    if int(os.environ.get('SMOKE_TEST', 0)):
        args.save_freq = 1

    return args


def prepare_imagenet(args, data_config):

    train_data_config = dict(**data_config)
    valid_data_config = data_config

    train_transform = create_transform(is_training=True, **train_data_config)
    train_transform = utils.add_inception_lighting(train_transform)
    train_dataset = ImageNet(args.data_root, split='train', transform=train_transform)

    valid_transform = create_transform(is_training=False, **valid_data_config)
    valid_dataset = ImageNet(args.data_root, split='val', transform=valid_transform)

    if int(os.environ.get('SMOKE_TEST', 0)):
        test_len = args.batch_size * 8
        train_dataset = torch.utils.data.Subset(
            train_dataset, indices=torch.randperm(len(train_dataset))[:test_len]
        )
        valid_dataset = torch.utils.data.Subset(
            valid_dataset, indices=torch.randperm(len(valid_dataset))[:test_len]
        )

    num_workers = args.workers

    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
    )
    valid_sampler = torch.utils.data.DistributedSampler(
        valid_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        sampler=valid_sampler,
        pin_memory=True,
        num_workers=num_workers
    )

    return train_loader, valid_loader


def main(args):

    log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stderr)

    # Distributed, device, PRNG
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.distributed = args.world_size > 1

    if args.distributed:
        logging.info('[dist] Distributed: wait dist process group:%d', args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method='env://',
                                world_size=int(os.environ['WORLD_SIZE']))
        assert args.world_size == dist.get_world_size()
        args.rank = dist.get_rank()
        logging.info('[dist] Distributed: success device:%d (%d/%d)', args.local_rank, args.rank, args.world_size)
    else:
        args.local_rank = 0
        args.rank = 0
        logging.info('[dist] Single processed')

    if torch.cuda.is_available():
        device = torch.device('cuda', args.local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    utils.set_random_seed_all(args.seed)

    writer = None
    if args.rank == 0:
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = os.path.join(args.save_dir, now)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        writer = SummaryWriter(args.save_dir)

    # Model
    model = create_model(args.model).to(device)
    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    if args.distributed and args.model == MODELS.EFFNET:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        model = NativeDDP(model, device_ids=[device], output_device=device)

    if args.local_rank == 0:
        logging.info(f'Model {safe_model_name(args.model)} created, '
                     f'param count:{sum([m.numel() for m in model.parameters()]) // 1024 // 1024:.2f}M')
        logging.info(model)

    # Data loaders
    train_loader, valid_loader = prepare_imagenet(args, data_config)

    if args.local_rank == 0:
        logging.info(f'Data loader created, '
                     f'batch size: {args.batch_size}x{args.world_size}, '
                     f'len (train): {len(train_loader)}, '
                     f'len (valid): {len(valid_loader)}')

    # Loss function
    if args.model == MODELS.EFFNET:
        loss_fn = LabelSmoothingCrossEntropy().to(device)
    else:
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
    loss_fn_eval = torch.nn.CrossEntropyLoss().to(device)

    # Optimizer
    optimizer = AdamP(model.parameters(),
                      delta=0.1, wd_ratio=0.1, weight_decay=args.weight_decay)

    if args.local_rank == 0:
        logging.info(f'optimizer created: {optimizer}')

    # Scheduler
    sched_config = dict(steps_per_epoch=len(train_loader), total_epochs=args.epochs)
    if args.decay_sched == 'cos':
        sched_config.update(immediate_cooldown=True, cooldown_type='cosine')
    else:
        sched_config.update(immediate_cooldown=False, cooldown_fraction=0.2, cooldown_type='cosine')
    scheduler = AutoWU(optimizer, device=device, **sched_config)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Training body
    if args.local_rank == 0:
        logging.info('Beginning of training.')

    final_metrics = {}
    for epoch_idx in range(args.epochs):
        train_metrics = train_one_epoch(args, epoch_idx, train_loader, model, loss_fn,
                                        scaler, optimizer, scheduler, device)
        valid_metrics = validate(args, epoch_idx, valid_loader, model, loss_fn_eval, device)

        if writer:
            with torch.no_grad():
                param_norm = torch.stack([p.pow(2.).sum() for p in model.parameters()]).sum().pow(0.5)

            writer.add_scalar('train/|p|', param_norm, epoch_idx + 1)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch_idx + 1)
            for key, value in train_metrics.items():
                writer.add_scalar(f'train/{key}', value, epoch_idx + 1)
            for key, value in valid_metrics.items():
                writer.add_scalar(f'valid/{key}', value, epoch_idx + 1)

        save = (epoch_idx+1) % args.save_freq == 0 or epoch_idx+1 == args.epochs
        if args.rank == 0 and save:
            model_path = os.path.join(args.save_dir, f'model.pth')
            torch.save(
                {
                    'epoch_idx': epoch_idx,
                    'model': model.state_dict(),
                    'metrics': valid_metrics,
                },
                model_path,
            )
            logging.info(f'[state] model saved to {model_path}')

        final_metrics = {**valid_metrics}

    if args.distributed:
        dist.barrier()

    if args.local_rank == 0:
        loss_final, acc1_final, acc5_final = final_metrics['loss'], final_metrics['acc1'], final_metrics['acc5']
        logging.info(f'[valid] final model, {args.epochs:d}, '
                     f'loss, {loss_final:.4f}, '
                     f'acc1, {acc1_final:.2f}, '
                     f'acc5, {acc5_final:.2f}')
        logging.info('End of training.')


def train_one_epoch(args, epoch_idx, train_loader, model, loss_fn, scaler, optimizer, scheduler, device):

    model.train()

    loss_meter = utils.AverageMeter(device=device)
    acc1_meter = utils.AverageMeter(device=device)
    acc5_meter = utils.AverageMeter(device=device)

    tic = time.time()
    torch.cuda.synchronize()

    for batch_idx, (input, target) in enumerate(train_loader):

        model.zero_grad(set_to_none=True)

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(input)
            loss = loss_fn(output, target)
            with torch.no_grad():
                acc1, acc5 = timm.utils.accuracy(output, target, topk=(1, 5))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss = loss.detach().clone()
        if args.distributed:
            utils.all_reduce(loss, reduction='mean')

        loss_meter.update(loss)
        acc1_meter.update(acc1)
        acc5_meter.update(acc5)

        scheduler.step(loss)

        epoch_pct = int((batch_idx + 1) / len(train_loader) * 100.0)
        log = epoch_pct > int(batch_idx / len(train_loader) * 100.0)
        if log and args.local_rank == 0:
            toc = time.time()
            lr = scheduler.get_last_lr()[0]
            logging.info(f'[train] epoch: {epoch_idx:03d}, '
                         f'{epoch_pct:3d}%, '
                         f'lr: {lr:.3e}, '
                         f'time elapsed: {toc-tic:.1f}s, '
                         f'loss: {loss_meter.average:.4f}, '
                         f'acc1: {acc1_meter.average:.2f}')

    torch.cuda.synchronize()
    toc = time.time()

    if args.local_rank == 0:
        logging.info(f'[train] epoch: {epoch_idx:03d}, '
                     f'time elapsed: {toc-tic:.1f}s, '
                     f'loss: {loss_meter.average:.4f}')
        metrics = {'loss': loss_meter.average, 'acc1': acc1_meter.average, 'acc5': acc5_meter.average}
    else:
        metrics = {}

    return metrics


@torch.no_grad()
def validate(args, epoch_idx, valid_loader, model, loss_fn, device):

    model.eval()

    loss_meter = utils.AverageMeter(device=device)
    acc1_meter = utils.AverageMeter(device=device)
    acc5_meter = utils.AverageMeter(device=device)

    tic = time.time()
    torch.cuda.synchronize()

    for batch_idx, (input, target) in enumerate(valid_loader):

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(input)
            loss = loss_fn(output, target)
            acc1, acc5 = timm.utils.accuracy(output, target, topk=(1, 5))

        if args.distributed:
            utils.all_reduce(loss, reduction='mean')
            utils.all_reduce(acc1, reduction='mean')
            utils.all_reduce(acc5, reduction='mean')

        batch_size = input.size(0)
        loss_meter.update(loss, n=batch_size)
        acc1_meter.update(acc1, n=batch_size)
        acc5_meter.update(acc5, n=batch_size)

        epoch_pct = int((batch_idx + 1) / len(valid_loader) * 100.0)
        log = epoch_pct > int(batch_idx / len(valid_loader) * 100.0)
        if log and args.local_rank == 0:
            toc = time.time()
            logging.info(f'[valid] epoch: {epoch_idx:03d}, '
                         f'{epoch_pct:3d}%, '
                         f'time elapsed: {toc - tic:.1f}s, '
                         f'loss: {loss_meter.average:.4f}, '
                         f'acc1: {acc1_meter.average:.2f}, '
                         f'acc5: {acc5_meter.average:.2f}')

    torch.cuda.synchronize()
    toc = time.time()

    if args.local_rank == 0:
        logging.info(f'[valid] epoch: {epoch_idx:03d}, '
                     f'time elapsed: {toc - tic:.1f}s, '
                     f'loss: {loss_meter.average:.4f}, '
                     f'acc1: {acc1_meter.average:.2f}, '
                     f'acc5: {acc5_meter.average:.2f}')

        metrics = {'loss': loss_meter.average, 'acc1': acc1_meter.average, 'acc5': acc5_meter.average}
    else:
        metrics = {}

    return metrics


if __name__ == '__main__':
    args = parse_args()
    main(args)
