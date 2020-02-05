import argparse
from datetime import time

import mltool_backend.utils.nn.utils as nn_utils
import torch

from ResNetEmbed import ResNetEmbed

from dataloader import get_test_set
from tqdm import tqdm
from torchsummary import summary
import statistics
from torch.utils.tensorboard import SummaryWriter
from opt import opt


def select_device(gpu_device):
    if (gpu_device == 'cpu') or (gpu_device == '-1') or (gpu_device == -1):
        device_to_use = '/cpu:0'
    else:
        device_to_use = '/gpu:' + str(gpu_device)

    nn_utils.select_device(gpu_device)



def train(train_loader, model, criterion, optimizer, level, criterion_additional, writer):
    top1_L1 = AverageMeter()
    top1_L2 = AverageMeter()
    top1_L3 = AverageMeter()
    top1_L4 = AverageMeter()

    topLoss_L1 = AverageMeter()
    topLoss_L2 = AverageMeter()
    topLoss_L3 = AverageMeter()
    topLoss_L4 = AverageMeter()

    model.train()

    T = 0.0625

    for i, (input, gt_family, gt_subfamily, gt_genus, gt_species) in enumerate(tqdm(train_loader)):
        input = input.cuda().requires_grad_()
        gt_family = gt_family.cuda()
        gt_subfamily = gt_subfamily.cuda()
        gt_genus = gt_genus.cuda()
        gt_species = gt_species.cuda()

        # compute output
        pred1_L1, pred1_L2, pred1_L3, pred1_L4, pred2_L2, pred2_L3, pred2_L4 = model(input)

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

        loss_L2 = loss_L2 + T * criterion_additional(pred2_L2, pred1_L2)
        loss_L3 = loss_L3 + T * criterion_additional(pred2_L3, pred1_L3)
        loss_L4 = loss_L4 + T * criterion_additional(pred2_L4, pred1_L4)
        total_loss = loss_L4 + loss_L3 + loss_L2 + loss_L1

        optimizer.zero_grad()
        if level == 1:
            loss_L1.backward()
        elif level == 2:
            loss_L2.backward()
        elif level == 3:
            loss_L3.backward()
        elif level == 4:
            loss_L4.backward()
        elif level == 5:
            total_loss.backward()
        optimizer.step()

        topLoss_L1.update(loss_L1.item(), input.size(0))
        topLoss_L2.update(loss_L2.item(), input.size(0))
        topLoss_L3.update(loss_L3.item(), input.size(0))
        topLoss_L4.update(loss_L4.item(), input.size(0))

        opt.trainIters +=1

        # Tensorboard
        writer.add_scalar('Train/Loss1', topLoss_L1.avg, opt.trainIters)
        writer.add_scalar('Train/Loss2', topLoss_L2.avg, opt.trainIters)
        writer.add_scalar('Train/Loss3', topLoss_L3.avg, opt.trainIters)
        writer.add_scalar('Train/Loss4', topLoss_L4.avg, opt.trainIters)
        writer.add_scalar('Train/LossTotal', total_loss, opt.trainIters)
        writer.add_scalar('Train/Acc1', top1_L1.avg, opt.trainIters)
        writer.add_scalar('Train/Acc2', top1_L2.avg, opt.trainIters)
        writer.add_scalar('Train/Acc3', top1_L3.avg, opt.trainIters)
        writer.add_scalar('Train/Acc4', top1_L4.avg, opt.trainIters)

    return top1_L1.avg, top1_L2.avg, top1_L3.avg, top1_L4.avg, topLoss_L1.avg, topLoss_L2.avg, topLoss_L3.avg, topLoss_L4.avg, total_loss


def valid(valid_loader, model, criterion, writer):
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
            pred1_L1, pred1_L2, pred1_L3, pred1_L4, _, _, _ = model(input)

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
        total_loss = loss_L4 + loss_L3 + loss_L2 + loss_L1

        opt.valIters += 1

        # Tensorboard
        writer.add_scalar('Val/Loss1', topLoss_L1.avg, opt.valIters)
        writer.add_scalar('Val/Loss2', topLoss_L2.avg, opt.valIters)
        writer.add_scalar('Val/Loss3', topLoss_L3.avg, opt.valIters)
        writer.add_scalar('Val/Loss4', topLoss_L4.avg, opt.valIters)
        writer.add_scalar('Val/LossTotal', total_loss, opt.valIters)
        writer.add_scalar('Val/Acc1', top1_L1.avg, opt.valIters)
        writer.add_scalar('Val/Acc2', top1_L2.avg, opt.valIters)
        writer.add_scalar('Val/Acc3', top1_L3.avg, opt.valIters)
        writer.add_scalar('Val/Acc4', top1_L4.avg, opt.valIters)

    return top1_L1.avg, top1_L2.avg, top1_L3.avg, top1_L4.avg, topLoss_L1.avg, topLoss_L2.avg, topLoss_L3.avg, topLoss_L4.avg, total_loss


def main():

    writer = SummaryWriter('runs/HSE_exp_2')

    # Create dataloader
    print("==> Creating dataloader...")
    data_dir = 'data/Butterfly200/images'
    val_list = 'data/Butterfly200/Butterfly200_val_release.txt'
    train_list = 'data/Butterfly200/Butterfly200_train_release.txt'

    val_loader = get_test_set(data_dir, val_list, opt)
    train_loader = get_test_set(data_dir, train_list, opt)

    classes_dict = {'family': 5, 'subfamily': 23, 'genus': 116, 'species': 200}

    # load the network
    print("==> Loading the network ...")
    model = ResNetEmbed(cdict=classes_dict)

    model.cuda()
    summary(model, (3, opt.crop_size, opt.crop_size))

    criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion_additional = torch.nn.KLDivLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    loss_decay = []
    level = 1
    model = torch.nn.DataParallel(model).cuda()

    for epoch in range(opt.num_epocs):
        print('############# Starting Epoch {} #############'.format(epoch))

        if level >= 2:
            for params in model.module.branch_L1.parameters():
                params.requires_grad = False
            for params in model.module.fc_L1.parameters():
                params.requires_grad = False
        if level >= 3:
            for params in model.module.branch_L2_guide.parameters():
                params.requires_grad = False
            for params in model.module.branch_L2_raw.parameters():
                params.requires_grad = False
            for params in model.module.fc_L2_raw.parameters():
                params.requires_grad = False
            for params in model.module.fc_L2_guide.parameters():
                params.requires_grad = False
            for params in model.module.G12.parameters():
                params.requires_grad = False
            for params in model.module.fc_L2_cat.parameters():
                params.requires_grad = False
        if level >= 4:
            for params in model.module.branch_L3_guide.parameters():
                params.requires_grad = False
            for params in model.module.branch_L3_raw.parameters():
                params.requires_grad = False
            for params in model.module.fc_L3_raw.parameters():
                params.requires_grad = False
            for params in model.module.fc_L3_guide.parameters():
                params.requires_grad = False
            for params in model.module.G23.parameters():
                params.requires_grad = False
            for params in model.module.fc_L3_cat.parameters():
                params.requires_grad = False
        if level == 5:
            for params in model.module.parameters():
                params.requires_grad = True
        for params in model.module.trunk.parameters():
            params.requires_grad = False
        acc1, acc2, acc3, acc4, loss1, loss2, loss3, loss4, total_loss = train(train_loader, model, criterion, optimizer, level,
                                                                   criterion_additional, writer)

        print(
            'Train-{idx:d} level:{level:.4f} | epoch | loss1:{loss1:.4f} | loss2:{loss2:.4f} | loss3:{loss3:.4f} | '
            'loss4:{loss4:.4f} | total_loss:{total_loss:.4f} | acc1:{acc1:.4f} | acc2:{acc2:.4f} | acc3:{acc3:.4f} | '
            'acc4:{acc4:.4f}'.format(
                idx=epoch, level=level,
                loss1=loss1, loss2=loss2, loss3=loss3, loss4=loss4, total_loss = total_loss,
                acc1=acc1, acc2=acc2, acc3=acc3, acc4=acc4
            ))
        acc1, acc2, acc3, acc4, loss1, loss2, loss3, loss4, total_loss = valid(val_loader, model, criterion, writer)
        print(
            'Valid-{idx:d} level:{level:.4f} | epoch | loss1:{loss1:.4f} | loss2:{loss2:.4f} | loss3:{loss3:.4f} | '
            'loss4:{loss4:.4f} | total_loss:{total_loss:.4f} | acc1:{acc1:.4f} | acc2:{acc2:.4f} | acc3:{acc3:.4f} | '
            'acc4:{acc4:.4f}'.format(
                idx=epoch, level=level,
                loss1=loss1, loss2=loss2, loss3=loss3, loss4=loss4, total_loss=total_loss,
                acc1=acc1, acc2=acc2, acc3=acc3, acc4=acc4
            ))
        if level == 1:
            loss_decay.append(loss1)
            scheduler.step(loss1)
        elif level == 2:
            loss_decay.append(loss2)
            scheduler.step(loss2)
        elif level == 3:
            loss_decay.append(loss3)
            scheduler.step(loss3)
        elif level == 4:
            loss_decay.append(loss4)
            scheduler.step(loss4)
        if len(loss_decay) > 5:
            loss_decay.pop(0)
            if statistics.stdev(loss_decay) < opt.standar_deviation:
                level += 1
                loss_decay = []
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
                if level == 5:
                    optimizer.param_groups[0]['lr'] == 0.0001

    m_module = model.module
    torch.save(m_module.state_dict(), 'hse.pth')

    writer.close()


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
    select_device(3)
    main()
