import argparse

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
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-l', '--learning_rate', default=0.1, type=float,
                        help='LR')
    parser.add_argument('--snapshot', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--crop_size', dest='crop_size', default=224, type=int,
                        help='crop size')
    parser.add_argument('--scale_size', dest='scale_size', default=448, type=int,
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
    model.eval()
    return 0, 0


def valid(valid_loader, model, criterion, optimizer):
    return 0, 0


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

    criterion = torch.nn.MSELoss().cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.l
    )
    for epoch in range(args.n):
        print('############# Starting Epoch {} #############'.format(epoch))
        loss, acc = train(train_loader, model, criterion, optimizer)

        print('Train-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            idx=epoch,
            loss=loss,
            acc=acc
        ))
        loss, acc = valid(val_loader, model, criterion, optimizer)
        print('Valid-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            idx=epoch,
            loss=loss,
            acc=acc
        ))
    m_module = model.module


if __name__ == '__main__':
    select_device(2)
    main()
