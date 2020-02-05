
import argparse

parser = argparse.ArgumentParser(description='PyTorch CabinMonitoringV1 Training')

parser.add_argument('--trainIters', default=0, type=int,
                    help='Total train iters')
parser.add_argument('--valIters', default=0, type=int,
                    help='Total valid iters')

parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-l', '--learning_rate', default=0.001, type=float,
                    help='LR')
parser.add_argument('-m', '--momentum', default=0.9, type=float,
                    help='Momentum')
parser.add_argument('-w', '--weight_decay', default=0.00005, type=float,
                    help='WD')
parser.add_argument('--snapshot', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--crop_size', dest='crop_size', default=448, type=int,
                    help='crop size')
parser.add_argument('--scale_size', dest='scale_size', default=512, type=int,
                    help='the size of the rescale image')
parser.add_argument('-n', '--num_epocs', default=100, type=int, help='Num epocs (default: 100)')
parser.add_argument('--standar_deviation', default=0.01, type=float, help='standar deviation of error used for change between levels')



opt = parser.parse_args()