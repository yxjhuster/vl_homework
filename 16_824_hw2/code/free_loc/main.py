import argparse
import os
import shutil
import time
import sys
sys.path.insert(0, 'faster_rcnn')
import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tensorboardX import SummaryWriter
from datetime import datetime
import cv2
import math

from datasets.factory import get_imdb
from custom import *

import pdb

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=32,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.01,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')
parser.add_argument('--log-dir', type=str, default='./tensorboardx',
                    help='Path to training data storage')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # DONE:
    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # set random seed
    torch.manual_seed(seed=1)
    np.random.seed(seed=1)

    # Data loading code
    # DONE: Write code for IMDBDataset in custom.py
    trainval_imdb = get_imdb('voc_2007_trainval')
    test_imdb = get_imdb('voc_2007_test')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = IMDBDataset(
        trainval_imdb,
        transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(
            test_imdb,
            transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    # DONE: Create loggers for visdom and tboard
    # DONE: You can pass the logger objects to train(), make appropriate
    # modifications to train()
    args.vis = True
    if args.vis:
        import visdom
        logdir = os.path.join(args.log_dir,
                              datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        writer = SummaryWriter(logdir)
        vis = visdom.Visdom(port='6009')

    if args.evaluate:
        validate(val_loader, model, criterion, writer, vis)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer, vis)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion,
                              writer, vis)
            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
            writer.add_scalar('validating/metric1', m1, epoch)
            writer.add_scalar('validating/metric1', m2, epoch)

    _, _ = validate(val_loader, model, criterion, writer, vis, plot=True)


# DONE: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, writer, vis):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    classes = train_loader.dataset.classes
    for i, (input, target) in enumerate(train_loader):
        iteration = epoch * len(train_loader) + i
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input.cuda().float()
        target_var = target.cuda().float()

        # DONE: Get output from model
        # DONE: Perform any necessary functions on the output
        # DONE: Compute loss using ``criterion``
        feat = model(input_var)

        # output resolution 29*29
        maxpooling = nn.MaxPool2d(kernel_size=(feat.size()[2], feat.size()[3]))
        output = maxpooling(feat)
        output = output.view(output.shape[0], 20)

        imoutput = nn.Sigmoid()(output)
        loss = criterion(imoutput, target_var)

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        # DONE:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        # TODO: Visualize things as mentioned in handout
        # TODO: Visualize at appropriate intervals
        writer.add_scalar('training/loss', loss.item(), iteration)
        writer.add_scalar('training/metric1', avg_m1.avg, iteration)
        writer.add_scalar('training/metric2', avg_m2.avg, iteration)
        if i % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(
                    name, param.clone().cpu().data.numpy(), iteration)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if (epoch%10==0 or epoch==args.epochs-1) and i%int(len(train_loader)/4)==0:
            img1 = input.cpu().numpy()[0]
            img2 = input.cpu().numpy()[1]

            img1 = np.array([img1[i] * std[i] + mean[i] for i in range(3)])
            img2 = np.array([img2[i] * std[i] + mean[i] for i in range(3)])

            # pred_label = imoutput.detach().cpu().numpy()

            gt_label = target.detach().cpu().numpy()

            # index = np.array([np.where(pred_label[j] == np.max(
            #     pred_label[j])) for j in range(pred_label.shape[0])])

            index1 = np.nonzero(gt_label[0])[0][0]
            index2 = np.nonzero(gt_label[1])[0][0]

            feature = feat.detach().cpu().numpy()

            heat1 = feature[0][index1]
            heat2 = feature[1][index2]

            # normalization
            heat1 = (heat1 - heat1.min()) / (heat1.max() - heat1.min())
            heat2 = (heat2 - heat2.min()) / (heat2.max() - heat2.min())

            # heat1 = heat1[0][0]
            # heat2 = heat2[0][0]

            img_heat1 = heat1 / np.max(heat1) * 255
            img_heat2 = heat2 / np.max(heat2) * 255
            img_heat1 = img_heat1.astype(np.uint8)
            img_heat2 = img_heat2.astype(np.uint8)

            img_heat1 = cv2.resize(img_heat1, (512, 512))
            img_heat2 = cv2.resize(img_heat2, (512, 512))

            img_heat1 = cv2.applyColorMap(img_heat1, cv2.COLORMAP_JET)
            img_heat2 = cv2.applyColorMap(img_heat2, cv2.COLORMAP_JET)

            img_heat1 = np.transpose(img_heat1 / 255., (2, 0, 1))
            img_heat2 = np.transpose(img_heat2 / 255., (2, 0, 1))

            writer.add_image('train/image1', img1, iteration)
            writer.add_image('train/heatmap1', img_heat1, iteration)
            writer.add_image('train/image2', img2, iteration)
            writer.add_image('train/heatmap2', img_heat2, iteration)
            if epoch % 2 == 0:
                vis.image(img1, opts=dict(title=str(epoch) + '_' +
                                          str(iteration) + '_' + str(1) + '_image'))
                vis.image(img_heat1, opts=dict(title=str(epoch) + '_' +
                                               str(iteration) + '_' + str(1) + '_heatmap_' + classes[index1]))
                vis.image(img2, opts=dict(title=str(epoch) + '_' +
                                          str(iteration) + '_' + str(2) + '_image'))
                vis.image(img_heat2, opts=dict(title=str(epoch) + '_' +
                                               str(iteration) + '_' + str(1) + '_heatmap_' + classes[index2]))

        # End of train()


def validate(val_loader, model, criterion, writer, vis, plot=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    classes = val_loader.dataset.classes
    for i, (input, target) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input.cuda().float()
        target_var = target.cuda().float()

        # DONE: Get output from model
        # DONE: Perform any necessary functions on the output
        # DONE: Compute loss using ``criterion``
        feat = model(input_var)
        maxpooling = nn.MaxPool2d(kernel_size=(feat.size()[2], feat.size()[3]))
        output = maxpooling(feat)
        output = output.view(output.shape[0], 20)
        imoutput = F.sigmoid(output)
        loss = criterion(imoutput, target_var)

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        # TODO: Visualize things as mentioned in handout
        # TODO: Visualize at appropriate intervals

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        batch_size = input.size()[0]

        if plot and i == 0:
            index = np.random.randint(batch_size, size=20)
            for idx in index:
                img = input.cpu().numpy()[idx]
                img = np.array([img[j] * std[j] + mean[j] for j in range(3)])
                gt_label = target.detach().cpu().numpy()[idx]
                feature = feat.detach().cpu().numpy()[idx]
                gt_idx = np.nonzero(gt_label)[0][0]
                heat1 = feature[gt_idx]
                heat1 = (heat1 - heat1.min()) / (heat1.max() - heat1.min())
                img_heat1 = heat1 / np.max(heat1) * 255
                img_heat1 = img_heat1.astype(np.uint8)
                img_heat1 = cv2.resize(img_heat1, (512, 512))
                img_heat1 = cv2.applyColorMap(img_heat1, cv2.COLORMAP_JET)
                img_heat1 = np.transpose(img_heat1 / 255., (2, 0, 1))
                vis.image(img, opts=dict(
                    title='validate_' + str(idx) + '_image'))
                vis.image(img_heat1, opts=dict(title='validate_' +
                                               str(idx) + '_heatmap_' + classes[gt_idx]))

    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed
    batch_size = output.size()[0]
    output_val = output.cpu().numpy().astype(np.float32)
    target_val = target.cpu().numpy().astype(np.float32)
    AP = []
    for i in range(batch_size):
        pred_cls = output_val[i]
        gt_cls = target_val[i]

        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=None)
        if math.isnan(ap):
            ap = 0
        AP.append(ap)
    mAP = np.nanmean(AP)
    return [mAP]


def metric2(output, target):
    # TODO: Ignore for now - proceed till instructed
    # using TOP5 inetead of TOP1
    batch_size = target.size(0)
    n_class = target.size(1)
    AP = 0.0
    for i in range(batch_size):
        pred_cls = output[i]
        idx = sorted(range(n_class),
                     key=lambda j: pred_cls[j], reverse=True)
        tmp = 0.0
        for k in range(5):
            tmp += output[i, idx[k]] * target[i, idx[k]]
        if tmp != 0:
            AP += 1.0
    return [AP / batch_size]


if __name__ == '__main__':
    main()
