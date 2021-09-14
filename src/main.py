"""
main.py
Autor: KyoungchanPark, EuisukChung, HyeongwonKang
예시 : python main.py
"""

from utils.progress import printProgressBar
from config import load_config, str2bool
from utils.check_mail import send_mail
from utils.utils import set_seed
from dataset import get_dataloader
import pandas as pd
import torch
import numpy as np
import random
import pickle
import warnings
import logging
import yaml
import sys
import os
import pdb

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, config):
    # arg parser
    model_name = args['model']

    # yaml parser
    data_path = config['data_path']
    processed_dataset_path = config['processed_dataset_path']
    WINDOW_GIVEN = config['window_given']
    WINDOW_SIZE = config['window_size']
    stride = config['stride']
    loader_params = config['loader_params']

    set_seed(72)

    trainloader, validloader, testloader = get_dataloader(data_path, processed_dataset_path, 
                                                          WINDOW_GIVEN, WINDOW_SIZE,
                                                          loader_params, train_stride=stride)

    pdb.set_trace()

if __name__ == '__main__':
    # logger 세팅
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('./log/HAICON.log')
    formatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # arg parser
    args = load_config()

    # yaml file
    with open('config.yaml') as f:
        config = yaml.load(f)

    # main
    try:
        main(args, config)
    except:
        logger.exception("error")
