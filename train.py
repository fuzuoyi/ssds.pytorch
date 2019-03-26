from __future__ import print_function

import sys
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import numpy as np
if '/data/software/opencv-3.4.0/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.4.0/lib/python2.7/dist-packages')
if '/data/software/opencv-3.3.1/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.3.1/lib/python2.7/dist-packages')
import cv2
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from lib.utils.config_parse import cfg_from_file
from lib.ssds_train import train_model

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
    parser.add_argument('--cfg', dest='config_file',
            help='optional config file', default=None, type=str)

    # for debug
    # sys.argv.append('--cfg')
    # sys.argv.append('./experiments/cfgs/ssd_xbase_train_voc.yml')
    # sys.argv.append('./experiments/cfgs/ssd_vgg16_train_voc.yml')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def train():
    args = parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    train_model()

if __name__ == '__main__':
    train()
