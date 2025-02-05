import os
import sys
import time
import torch
import random
import logging
import argparse
from datetime import datetime

def get_config():
    parser = argparse.ArgumentParser()

    # Base configuration
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='ResNet50',
                        choices=['AlexNet', 'GoogleNet', 'VGG16', 'ResNet50'])

    # Optimization parameters
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # Environment settings
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backend', default=False, action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--index', type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    # Logger setup
    args.log_name = '{}_{}.log'.format(args.model_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    log_dir = 'logs'
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
    return args, logger
