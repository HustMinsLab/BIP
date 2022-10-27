# -*- coding: utf-8 -*-

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='ACE')

    # # training arguments
    parser.add_argument('--seed',         default=209,      type=int, help='seed for reproducibility')
    parser.add_argument('--batch_size',   default=4,       type=int, help='batchsize for optimizer updates')
    parser.add_argument('--batch_num',   default=1,       type=int, help='batchnum for optimizer updates')
    parser.add_argument('--wd',           default=1e-2,     type=float, help='weight decay')  # 1e-3

    parser.add_argument('--num_epoch',    default=20,       type=int, help='number of total epochs to run')
    parser.add_argument('--lr',           default=1e-5,     type=float, help='initial learning rate')
    parser.add_argument('--warm_ratio',   default=0.1,      type=float, help='ratio of warm up')

    parser.add_argument('--file_out',  default='out',  type=str, help='Result file name')
    
    args = parser.parse_args()
    return args
