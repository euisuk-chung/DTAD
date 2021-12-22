"""
main.py
Autor: KyoungchanPark, EuisukChung, HyeongwonKang
예시 : python main.py
"""

from utils.progress import printProgressBar
from config import load_config
from utils.utils import set_seed, range_check, check_graph, put_labels, fill_blank
from dataset import get_dataloader
from model import Model
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
    type = args['type']
    GPU_NUM = args.gpu  # select gpu number
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)  # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device())  # check

    try:
        assert model_name in ('StackedGRU', 'MADGAN', 'TAnoGAN', 'LSTMAE', '1DCONVAE', 'Transformer_Encoder', 'GPT')
        assert type in ('Train', 'Test')
    except AssertionError as e:
        raise

    if model_name in ('MADGAN', 'TAnoGAN', 'LSTMAE', '1DCONVAE'):
        model_type = 'generative'
    else:
        model_type = 'predictive'

    # yaml parser
    data_path = config['data_path']
    processed_dataset_path = config['processed_dataset_path']
    WINDOW_GIVEN = config['window_given']
    WINDOW_SIZE = config['window_size']
    stride = config['stride']
    loader_params = config['loader_params']
    model_params = config[model_name]

    set_seed(72)


    trainloader, validloader, testloader = get_dataloader(data_path, processed_dataset_path, 
                                                          WINDOW_GIVEN, WINDOW_SIZE,
                                                          loader_params, model_type, train_stride=stride)

    model = Model(model_name, trainloader, validloader, config.epochs, criterion,
                optimizer, device, config.early_patience, config.early_verbose, model_params)

    if type == 'Train':
        model.train()

        CHECK_TS, CHECK_DIST, CHECK_ATT = model.validation()
        ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)

        if args.range_check != 0:
            ANOMALY_SCORE = range_check(ANOMALY_SCORE, size=args.range_check)

        graph = check_graph(ANOMALY_SCORE, CHECK_ATT, piece=2, THRESHOLD=args.threshold)
        graph.savefig(os.path.join(config.output_path, f'{model_name}_{config.epochs}.png'))

        LABELS = put_labels(ANOMALY_SCORE, THRESHOLD)
        ATTACK_LABELS = put_labels(np.array(VALIDATION_DF_RAW[ATTACK_FIELD]), threshold=args.threshold)
        FINAL_LABELS = fill_blank(CHECK_TS, LABELS, np.array(VALIDATION_DF_RAW[TIMESTAMP_FIELD]))

        TaPR = etapr.evaluate_haicon(anomalies=ATTACK_LABELS, predictions=FINAL_LABELS)
        logger.info(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
        logger.info(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
        logger.info(f"Detected anomalies: {TaPR['Detected_Anomalies']}")

    else:
        CHECK_TS, CHECK_DIST, CHECK_ATT = model.validation()
        ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)

        if args.range_check != 0:
            ANOMALY_SCORE = range_check(ANOMALY_SCORE, size=args.range_check)

        graph = check_graph(ANOMALY_SCORE, CHECK_ATT, piece=2, THRESHOLD=args.threshold)
        graph.savefig()

        LABELS = put_labels(ANOMALY_SCORE, THRESHOLD)
        ATTACK_LABELS = put_labels(np.array(VALIDATION_DF_RAW[ATTACK_FIELD]), threshold=args.threshold)
        FINAL_LABELS = fill_blank(CHECK_TS, LABELS, np.array(VALIDATION_DF_RAW[TIMESTAMP_FIELD]))

        TaPR = etapr.evaluate_haicon(anomalies=ATTACK_LABELS, predictions=FINAL_LABELS)
        logger.info(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
        logger.info(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
        logger.info(f"Detected anomalies: {TaPR['Detected_Anomalies']}")

    pdb.set_trace()

if __name__ == '__main__':
    # logger 세팅
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('./log/timesereis_anomaly.log')
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
