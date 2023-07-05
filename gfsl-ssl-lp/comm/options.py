
from __future__ import print_function

import os
import argparse
import socket
import time
import sys
import shutil

# import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import model_pool
from models.util import create_model

from models.criterion import DistillKL
from models.model import Classifier

from dataset.transform_cfg import transforms_options, transforms_list
from dataset.get_loader import get_train_loader, test_loader

# from util import *
from tqdm import tqdm
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=64, help='batch_size 50')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=80, help='number of training epochs')
    parser.add_argument('--seed', type=int, default=100)

    # optimization
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,70', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='CUB',
                        choices=['miniImageNet', 'Dog','CIFAR-FS', 'CUB', 'tieredImageNet'])
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--task_mode', type=str, default='gfsl')

    parser.add_argument('--method', type=str, default='baseline',
                        choices=['baseline', 'protonet', 'ours', 'distill', 'rfs', 'in-eq'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', default=False, help='use trainval set')


    # model parameter
    parser.add_argument('--fea_dim', type=int, default=640, help='feature dimension')
    parser.add_argument('--partion', default=1.0, type=float, help='radio of training instances')

    # specify folder
    parser.add_argument('--model_path', type=str, default='./save_ori', help='path to save model')
    # parser.add_argument('--tb_path', type=str, default='./tensorboard', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='/data/datasets', help='path to data root')
    parser.add_argument('--model_pretrained', type=str, default='./save1')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=300, help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5,
                        help='Number of classes for doing each classification run')
    # parser.add_argument('--n_shots', type=int, default=1,
    #                     help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15,
                        help='Number of query in test')
    parser.add_argument('--test_batch_size', type=int, default=1,help='Size of test batch)')
    parser.add_argument('--test_state', type=str, default='val')

    opt = parser.parse_args()
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'
    opt.n_aug_support_samples = 1

    # set the path according to the environment
    # if not opt.tb_path:
    #     opt.tb_path = './tensorboard'
    # if not opt.data_root:
    #     opt.data_root = './data/{}'.format(opt.dataset)
    # else:
    #     opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}'.format(opt.dataset, opt.method, opt.model)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name, opt.mode)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    opt.tb_path = os.path.join(opt.save_folder, 'tensorboard')
    if os.path.exists(opt.tb_path):
        shutil.rmtree(opt.tb_path)
    os.mkdir(opt.tb_path)

    opt.n_gpu = torch.cuda.device_count()
    return opt
