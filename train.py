import argparse
import os
import random
import shutil
import time
import warnings
import sys
import cv2

import numpy as np
import scipy.misc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from datasets import get_dataset
from models import get_classification_model

from sr_models.model import RDN, Vgg19

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data-root-pos', type=str, default='./data',
                    help='path to dataset')
parser.add_argument('--data-root-neg', type=str, default='./data',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='cityscapes',
                            help='dataset name (default: pascal12)')
parser.add_argument('-a', '--arch', type=str, default='resnet50',                   
                    help='model architecture')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--input-channel', default=3, type=int,
                    help='number of input channel')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--save-every-epoch', type=int, default=2,
                    help='how many epochs to save a model.')
parser.add_argument('--output-path', default='./output_models', type=str, metavar='PATH',
                    help='path to output models')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--dataset_type', type=str, default='image',
                    help='which dataset to load.')

parser.add_argument('--carlibration', default=1.0, type=float,
                    help='carlibration factor for posterior')
parser.add_argument('--defense', default=1.0, type=float,
                    help='defense factor')
parser.add_argument('--save_path', type=str, default='./score.npy', help='save models')

parser.add_argument('--no_dilation', action='store_true', help='do not use dilated convolutions in attackers')

parser.add_argument('--sr-num-features', type=int, default=64)
parser.add_argument('--sr-growth-rate', type=int, default=64)
parser.add_argument('--sr-num-blocks', type=int, default=16)
parser.add_argument('--sr-num-layers', type=int, default=8)
parser.add_argument('--sr-scale', type=int, default=4)

parser.add_argument('--sr-weights-file', type=str, required=True)

parser.add_argument('--idx-stages', type=int, default=0)

parser.add_argument('--lr-sr',  default=0.02, type=float, help=' learning rate for resolution')
parser.add_argument('--lw-sr', default=1, type=float, help='loss weight for reconstruction')
parser.add_argument('--mode-sr', default='none', type=str, help='can be none, colorization or denoising.')
parser.add_argument('--fixed-sr', dest='fixed_sr', action='store_true',
                    help='use fixed super resolution model')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    model = get_classification_model(arch=args.arch, pretrained = args.pretrained,
                                     input_channel=args.input_channel, num_classes=2, dilated=(not args.no_dilation))
                                     
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    criterion2 = nn.L1Loss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:%d'%(args.gpu))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    test_dataset = get_dataset(name=args.dataset_type, root_pos=args.data_root_pos, root_neg=args.data_root_neg, flip=True)

    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    sr_model = RDN(scale_factor=args.sr_scale,
                num_channels=3,
                num_features=args.sr_num_features,
                growth_rate=args.sr_growth_rate,
                num_blocks=args.sr_num_blocks,
                num_layers=args.sr_num_layers,
                requires_grad=False).cuda(args.gpu)#.to(device)
    
    checkpoint = torch.load(args.sr_weights_file, map_location='cuda:%d'%(args.gpu))
    if 'state_dict' in checkpoint.keys():
        sr_model.load_state_dict(checkpoint['state_dict'])
    else:
        sr_model.load_state_dict(checkpoint)
        
    perception_net = Vgg19().cuda(args.gpu)

    if args.fixed_sr:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(list(model.parameters()) + list(sr_model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    lr_epoch =  0.9999 * np.cos(np.pi / 2 * np.arange(args.epochs) / (args.epochs-1) ) + 0.0001

    for epoch in range( args.epochs ):
        # train for one epoch
        Precision, Recall, Score = train(epoch, test_loader, model, sr_model, perception_net, optimizer, criterion, criterion2, args)
        adjust_learning_rate(optimizer, lr_epoch[epoch] * args.lr )
    
        if epoch % args.save_every_epoch == 0 or epoch == args.epochs-1:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='%s/%04d.pth.tar'%(args.output_path , epoch+1))

            save_checkpoint({
                'state_dict': sr_model.state_dict(),
            }, is_best=False, filename='%s/%04d_sr.pth.tar'%(args.output_path , epoch+1))


def train(epoch, test_loader, model, sr_model, perception_net, optimizer, criterion, criterion2, args):
    TP = 0
    FP = 0
    FN = 0

    TOTAL = 0
    CORRECT = 0

    # switch to eval mode
    model.train()
    sr_model.train()

    score = []
    for i, (input, target, post_path) in enumerate(test_loader):     
         
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        lr = 0
        for ii in range(args.sr_scale):
            for jj in range(args.sr_scale):
                lr = lr + input[:, :, ii::args.sr_scale, jj::args.sr_scale] / (args.sr_scale * args.sr_scale)

        lr = lr / 255.0
        input = input / 255.0

        optimizer.zero_grad()

        preds_input = sr_model(lr)

        if args.idx_stages > 0:
            per_rec = perception_net(preds_input)
            per_gt = perception_net(input)
            rec_features = abs( per_rec[args.idx_stages - 1] - per_gt[args.idx_stages - 1] )

            output, aa = model( rec_features )
        else:
            rec_features0 = abs( preds_input - input )

            output, aa = model( rec_features0 )

        loss = criterion(output , target)

        # compute output
        loss2 = 0
        
        if sum(target==1) > 1 and args.fixed_sr == False:
            if np.random.uniform(0,1) > 0.5 and args.mode_sr == 'denoising':
                dims = lr.shape
                lr += torch.empty(lr.shape).normal_(mean=0,std=4.0 / 255).cuda()
            elif np.random.uniform(0,1) > 0.5 and args.mode_sr == 'colorization':
                dims = lr.shape
                mask = np.random.uniform(0,1,(dims[0],dims[2],dims[3]))
                mask = mask < np.random.uniform(0.1,0.25)
                tmp = lr.mean(dim=1)
                for i in range( dims[0] ):
                    lr[i,:,mask[i]] = tmp[i,mask[i]]
                    
            preds_input = sr_model(lr)
            per_rec = perception_net(preds_input)
            per_gt = perception_net(input)

            loss_rec_real = criterion2(preds_input[target==1], input[target==1])
        
            loss_rec_real_feat = criterion2(per_rec[0][target==1], per_gt[0][target==1])
        
            loss_rec_real_feat2 = criterion2(per_rec[1][target==1], per_gt[1][target==1])
    
            loss_rec_real_feat3 = criterion2(per_rec[2][target==1], per_gt[2][target==1])
        
            loss_rec_real_feat4 = criterion2(per_rec[3][target==1], per_gt[3][target==1])

            loss_rec_real_feat5 = criterion2(per_rec[4][target==1], per_gt[4][target==1])
            
            loss2 = loss_rec_real * args.lw_sr
            loss2 += (1.0 / 32) * (loss_rec_real_feat) * args.lw_sr
            loss2 += (1.0 / 16) * (loss_rec_real_feat2 ) * args.lw_sr
            loss2 += (1.0 / 8) * (loss_rec_real_feat3 ) * args.lw_sr
            loss2 += (1.0 / 4) * (loss_rec_real_feat4 ) * args.lw_sr
            loss2 += 1.0 * (loss_rec_real_feat5 ) * args.lw_sr

            loss_total = loss + loss2
        else:
            loss_total = loss

        loss_total.backward()

        for p in model.parameters():
            p.grad.mul_( args.lr_sr )

        optimizer.step()

        pred = (output[:,0] < output[:,1])
        CORRECT += pred.eq(target.view_as(pred)).sum().item()
        TOTAL += output.shape[0]

        
        TP = TP + sum((target==pred)*(1==pred))
        FP = FP + sum((target!=pred)*(1==pred))
        FN = FN + sum((target!=pred)*(0==pred))
        print('%d (%d/%d) : Precision=%.4f, Recall = %.4f, Accuracy,%.4f, Loss=%.4f, Loss_Rec=%.4f' % (epoch, i, len(test_loader), 1.0*TP/(TP+FP), 1.0*TP/(TP+FN), 1.0*CORRECT/TOTAL,loss, loss2 ))

    return 1.0*TP/(TP+FP), 1.0*TP/(TP+FN), np.array(score)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

