import argparse
from datetime import time

import mltool_backend.utils.nn.utils as nn_utils
import torch

from ResNetEmbed import ResNetEmbed

from dataloader import get_test_set
from tqdm import tqdm


def select_device(gpu_device):
    if (gpu_device == 'cpu') or (gpu_device == '-1') or (gpu_device == -1):
        device_to_use = '/cpu:0'
    else:
        device_to_use = '/gpu:' + str(gpu_device)

    nn_utils.select_device(gpu_device)


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch HSE deploying')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-l', '--learning_rate', default=0.001, type=float,
                        help='LR')
    parser.add_argument('--snapshot', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--crop_size', dest='crop_size', default=448, type=int,
                        help='crop size')
    parser.add_argument('--scale_size', dest='scale_size', default=512, type=int,
                        help='the size of the rescale image')
    parser.add_argument('-n', '--num_epocs', default=100, type=int, help='Num epocs (default: 100)')
    args = parser.parse_args()

    return args


def print_args(args):
    print("==========================================")
    print("==========       CONFIG      =============")
    print("==========================================")
    for arg, content in args.__dict__.items():
        print("{}:{}".format(arg, content))
    print("\n")


def train(train_loader, model, criterion, optimizer):
    top1_L1 = AverageMeter()
    top1_L2 = AverageMeter()
    top1_L3 = AverageMeter()
    top1_L4 = AverageMeter()

    topLoss_L1 = AverageMeter()
    topLoss_L2 = AverageMeter()
    topLoss_L3 = AverageMeter()
    topLoss_L4 = AverageMeter()

    model.train()

    for i, (input, gt_family, gt_subfamily, gt_genus, gt_species) in enumerate(tqdm(train_loader)):
        input = input.cuda().requires_grad_()
        gt_family = gt_family.cuda()
        gt_subfamily = gt_subfamily.cuda()
        gt_genus = gt_genus.cuda()
        gt_species = gt_species.cuda()

        # compute output
        pred1_L1, pred1_L2, pred1_L3, pred1_L4 = model(input)

        # measure accuracy and record loss
        prec1_L1, _ = accuracy(pred1_L1.data, gt_family, topk=(1, 5))
        prec1_L2, _ = accuracy(pred1_L2.data, gt_subfamily, topk=(1, 5))
        prec1_L3, _ = accuracy(pred1_L3.data, gt_genus, topk=(1, 5))
        prec1_L4, _ = accuracy(pred1_L4.data, gt_species, topk=(1, 5))

        top1_L1.update(prec1_L1, input.size(0))
        top1_L2.update(prec1_L2, input.size(0))
        top1_L3.update(prec1_L3, input.size(0))
        top1_L4.update(prec1_L4, input.size(0))

        loss_L1 = criterion(pred1_L1, gt_family)
        loss_L2 = criterion(pred1_L2, gt_subfamily)
        loss_L3 = criterion(pred1_L3, gt_genus)
        loss_L4 = criterion(pred1_L4, gt_species)

        topLoss_L1.update(loss_L1.item(), input.size(0))
        topLoss_L2.update(loss_L2.item(), input.size(0))
        topLoss_L3.update(loss_L3.item(), input.size(0))
        topLoss_L4.update(loss_L4.item(), input.size(0))

        #total_loss = loss_L4 + loss_L3 + loss_L2 + loss_L1

        optimizer.zero_grad()
        loss_L4.backward()
        optimizer.step()



    return top1_L1.avg, top1_L2.avg, top1_L3.avg, top1_L4.avg, topLoss_L1.avg, topLoss_L2.avg, topLoss_L3.avg, topLoss_L4.avg


def valid(valid_loader, model, criterion):
    top1_L1 = AverageMeter()
    top1_L2 = AverageMeter()
    top1_L3 = AverageMeter()
    top1_L4 = AverageMeter()

    topLoss_L1 = AverageMeter()
    topLoss_L2 = AverageMeter()
    topLoss_L3 = AverageMeter()
    topLoss_L4 = AverageMeter()

    model.eval()

    for i, (input, gt_family, gt_subfamily, gt_genus, gt_species) in enumerate(tqdm(valid_loader)):
        gt_family = gt_family.cuda()
        gt_subfamily = gt_subfamily.cuda()
        gt_genus = gt_genus.cuda()
        gt_species = gt_species.cuda()
        with torch.no_grad():
            # compute output
            pred1_L1, pred1_L2, pred1_L3, pred1_L4 = model(input)

            loss_L1 = criterion(pred1_L1, gt_family)
            loss_L2 = criterion(pred1_L2, gt_subfamily)
            loss_L3 = criterion(pred1_L3, gt_genus)
            loss_L4 = criterion(pred1_L4, gt_species)

        # measure accuracy and record loss
        prec1_L1, _ = accuracy(pred1_L1.data, gt_family, topk=(1, 5))
        prec1_L2, _ = accuracy(pred1_L2.data, gt_subfamily, topk=(1, 5))
        prec1_L3, _ = accuracy(pred1_L3.data, gt_genus, topk=(1, 5))
        prec1_L4, _ = accuracy(pred1_L4.data, gt_species, topk=(1, 5))

        top1_L1.update(prec1_L1, input.size(0))
        top1_L2.update(prec1_L2, input.size(0))
        top1_L3.update(prec1_L3, input.size(0))
        top1_L4.update(prec1_L4, input.size(0))

        topLoss_L1.update(loss_L1.item(), input.size(0))
        topLoss_L2.update(loss_L2.item(), input.size(0))
        topLoss_L3.update(loss_L3.item(), input.size(0))
        topLoss_L4.update(loss_L4.item(), input.size(0))

    return top1_L1.avg, top1_L2.avg, top1_L3.avg, top1_L4.avg, topLoss_L1.avg, topLoss_L2.avg, topLoss_L3.avg, topLoss_L4.avg


def main():
    args = arg_parse()
    print_args(args)

    # Create dataloader
    print("==> Creating dataloader...")
    data_dir = 'data/Butterfly200/images'
    val_list = 'data/Butterfly200/Butterfly200_val_release.txt'
    train_list = 'data/Butterfly200/Butterfly200_train_release.txt'

    val_loader = get_test_set(data_dir, val_list, args)

    train_loader = get_test_set(data_dir, train_list, args)

    classes_dict = {'family': 5, 'subfamily': 23, 'genus': 116, 'species': 200}

    # load the network
    print("==> Loading the network ...")
    model = ResNetEmbed(cdict=classes_dict)

    model.cuda()

    model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.00005)
    for epoch in range(args.num_epocs):
        print('############# Starting Epoch {} #############'.format(epoch))
        acc1, acc2, acc3, acc4, loss1, losss2, loss3, loss4 = train(train_loader, model, criterion, optimizer)

        print('Train-{idx:d} epoch | loss1:{loss1:.4f} | acc1:{acc1:.4f}'.format(
            idx=epoch,
            loss1=loss1,
            acc1=acc1
        ))
        acc1, acc2, acc3, acc4, loss1, losss2, loss3, loss4 = valid(val_loader, model, criterion)
        print('Valid-{idx:d} epoch | loss1:{loss1:.4f} | acc1:{acc1:.4f}'.format(
            idx=epoch,
            loss1=loss1,
            acc1=acc1
        ))
    m_module = model.module


class AverageMeter(object):
    """Computes and stores the   average and current value"""

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    select_device(2)
    main()
