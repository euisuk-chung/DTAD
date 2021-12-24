"""
main.py
Autor: KyoungchanPark, EuisukChung, HyeongwonKang
예시 : python main.py
"""

from config import load_config
from utils.utils import set_seed, range_check, check_graph, put_labels, fill_blank
from dataset import get_dataloader, dataframe_from_csvs
from model import Model
import torch
import numpy as np
from pathlib import Path
import warnings
import logging
import yaml
import os
from TaPR_pkg import etapr
from datetime import datetime


warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
now = datetime.now()

def main(args, config):
    # arg parser
    model_name = args['model']
    type = args['type']
    GPU_NUM = args['gpu']  # select gpu number
    checkpoint = args['checkpoint']
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)  # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device())  # check

    try:
        assert model_name in ('GRU', 'LSTMAE', 'CONV1dAE', 'TransformerEncoder', 'GPT')
        assert type in ('Train', 'Test')
    except AssertionError as e:
        raise

    if model_name in ('LSTMAE', 'CONV1dAE'):
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
    checkpoint_path = config['checkpoint_path']
    save_path = config['save_path']

    set_seed(72)

    TEST_DATASET = sorted([x for x in Path(os.path.join(data_path, 'test')).glob("test*.csv")])
    VALIDATION_DATASET = sorted([x for x in Path(os.path.join(data_path, 'validation')).glob("validation*.csv")])
    VALIDATION_DF_RAW = dataframe_from_csvs(VALIDATION_DATASET)
    TEST_DF_RAW = dataframe_from_csvs(TEST_DATASET)

    trainloader, validloader, testloader = get_dataloader(data_path, processed_dataset_path, 
                                                          WINDOW_GIVEN, WINDOW_SIZE,
                                                          loader_params, model_type, train_stride=stride)


    model = Model(model_name, trainloader, validloader, config['epochs'], device,
                  config['early_patience'], config['early_verbose'], checkpoint_path, model_params, checkpoint)

    if type == 'Train':
        logger.info('Training Start!!')
        model.train()

        CHECK_TS, CHECK_DIST, CHECK_ATT = model.validation()
        ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)

        if args['range_check'] != 0:
            ANOMALY_SCORE = range_check(ANOMALY_SCORE, size=args['range_check'])

        graph = check_graph(ANOMALY_SCORE, CHECK_ATT, piece=2, THRESHOLD=args['threshold'])
        graph.savefig(os.path.join(save_path, f'{model_name}_{config["epochs"]}.png'))

        LABELS = put_labels(ANOMALY_SCORE, args['threshold'])
        ATTACK_LABELS = put_labels(np.array(VALIDATION_DF_RAW["attack"]), threshold=args['threshold'])
        FINAL_LABELS = fill_blank(CHECK_TS, LABELS, np.array(VALIDATION_DF_RAW["timestamp"]))

        TaPR = etapr.evaluate_haicon(anomalies=ATTACK_LABELS, predictions=FINAL_LABELS)
        logger.info(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
        logger.info(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
        logger.info(f"Detected anomalies: {TaPR['Detected_Anomalies']}")

    else:
        CHECK_TS, CHECK_DIST, CHECK_ATT = model.validation()
        ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)

        if args['range_check'] != 0:
            ANOMALY_SCORE = range_check(ANOMALY_SCORE, size=args['range_check'])

        graph = check_graph(ANOMALY_SCORE, CHECK_ATT, piece=2, THRESHOLD=args['threshold'])
        graph.savefig(os.path.join(save_path, f'{model_name}_{now.strftime("%Y%m%d_%H%m%S")}.png'))

        LABELS = put_labels(ANOMALY_SCORE, args['threshold'])
        ATTACK_LABELS = put_labels(np.array(VALIDATION_DF_RAW["attack"]), threshold=args['threshold'])
        FINAL_LABELS = fill_blank(CHECK_TS, LABELS, np.array(VALIDATION_DF_RAW["timestamp"]))

        TaPR = etapr.evaluate_haicon(anomalies=ATTACK_LABELS, predictions=FINAL_LABELS)
        logger.info(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
        logger.info(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
        logger.info(f"Detected anomalies: {TaPR['Detected_Anomalies']}")

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
