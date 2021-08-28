import sys
import os
from pathlib import Path
from datetime import timedelta
import dateutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

def normalize(df, TAG_MIN, TAG_MAX):
    ndf = df.copy()
    for c in df.columns:
        if TAG_MIN[c] == TAG_MAX[c]:
            ndf[c] = df[c] - TAG_MIN[c]
        else:
            ndf[c] = (df[c] - TAG_MIN[c]) / (TAG_MAX[c] - TAG_MIN[c])
    return ndf

def boundary_check(df):
    x = np.array(df, dtype=np.float32)
    return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))

class HaiDataset(Dataset):
    def __init__(self, timestamps, df, WINDOW_GIVEN=89, WINDOW_SIZE=90, stride=1, attacks=None):
        self.ts = np.array(timestamps)
        self.tag_values = np.array(df, dtype=np.float32)
        self.WINDOW_GIVEN = WINDOW_GIVEN
        self.WINDOW_SIZE = WINDOW_SIZE
        self.valid_idxs = []
        for L in range(len(self.ts) - self.WINDOW_SIZE + 1):
            R = L + WINDOW_SIZE - 1
            if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(
                self.ts[L]
            ) == timedelta(seconds=self.WINDOW_SIZE - 1):
                self.valid_idxs.append(L)
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)
        print(f"# of valid windows: {self.n_idxs}")
        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
            self.with_attack = True
        else:
            self.with_attack = False

    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        last = i + self.WINDOW_SIZE - 1
        item = {"attack": self.attacks[last]} if self.with_attack else {}
        item["ts"] = self.ts[i + self.WINDOW_SIZE - 1]
        item["given"] = torch.from_numpy(self.tag_values[i : i + self.WINDOW_GIVEN])
        item["answer"] = torch.from_numpy(self.tag_values[last])
        return item

def get_normalized_data(data_path):
    TRAIN_DATASET = sorted([x for x in Path(os.path.join(data_path, 'train')).glob("train*.csv")])
    TEST_DATASET = sorted([x for x in Path(os.path.join(data_path, 'test')).glob("test*.csv")])
    VALIDATION_DATASET = sorted([x for x in Path(os.path.join(data_path, 'validation')).glob("validation*.csv")])

    TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)
    VALIDATION_DF_RAW = dataframe_from_csvs(VALIDATION_DATASET)
    TEST_DF_RAW = dataframe_from_csvs(TEST_DATASET)

    TIMESTAMP_FIELD = "timestamp"
    IDSTAMP_FIELD = 'id'
    ATTACK_FIELD = "attack"
    VALID_COLUMNS_IN_TRAIN_DATASET = TRAIN_DF_RAW.columns.drop([TIMESTAMP_FIELD])
    VALID_COLUMNS_IN_TRAIN_DATASET

    TAG_MIN = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].min()
    TAG_MAX = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].max()

    TRAIN_DF = normalize(TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET], TAG_MIN, TAG_MAX).ewm(alpha=0.9).mean()
    VALIDATION_DF = normalize(VALIDATION_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET], TAG_MIN,
                              TAG_MAX)  # 왜 validation은 exponential weighted function 안함?
    TEST_DF = normalize(TEST_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET], TAG_MIN, TAG_MAX).ewm(alpha=0.9).mean()

    return {'Train': {'timestamps': TRAIN_DF_RAW[TIMESTAMP_FIELD],
                      'dataframe': TRAIN_DF,
                      'attacks': None},
           'Valid': {'timestamps': VALIDATION_DF_RAW[TIMESTAMP_FIELD],
                     'dataframe': VALIDATION_DF,
                     'attacks': VALIDATION_DF_RAW[ATTACK_FIELD]},
            'Test': {'timestamps': TEST_DF_RAW[TIMESTAMP_FIELD],
                     'dataframe': TEST_DF,
                     'attacks': None}}

def check_datafile(data_path, processed_dataset_path, dataset_type, WINDOW_GIVEN, WINDOW_SIZE, stride=1):
    try:
        assert dataset_type in ('Train', 'Test', 'Valid'), f"k must be Train or Test or Valid"
    except AssertionError as e:
        raise

    path = os.path.join(processed_dataset_path, dataset_type,
                        f'{dataset_type}_Given{WINDOW_GIVEN}_Size{WINDOW_SIZE}_Stride{stride}.pkl')

    if os.path.isfile(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f'{path} load complete!')
    else:
        print(f'Creating {path}')
        normalized_data_path = os.path.join(processed_dataset_path, 'normalized_data.pkl')
        if os.path.isfile(normalized_data_path):
            with open(normalized_data_path, "rb") as f:
                normalized_data = pickle.load(f)
            print(f'{normalized_data_path} load complete!')
        else:
            print(f'Creating {normalized_data_path}')
            normalized_data = get_normalized_data(data_path)
            with open(normalized_data_path, "wb") as f:
                pickle.dump(normalized_data, f)
            print(f'{normalized_data_path} generation complete!')

        data = HaiDataset(
            normalized_data[dataset_type]['timestamps'], normalized_data[dataset_type]['dataframe'],
            WINDOW_GIVEN, WINDOW_SIZE, stride=stride, attacks=normalized_data[dataset_type]['attacks'])

        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f'{path} generation complete!')

    return data


def get_dataloader(data_path, processed_dataset_path, WINDOW_GIVEN, WINDOW_SIZE, dataloader_params, train_stride=10):
    # HAI_DATASET_TRAIN = HaiDataset(
    #     normalized_data['Train']['timestamps'], normalized_data['TRAIN']['dataframe'],
    #     WINDOW_GIVEN, WINDOW_SIZE, stride=10
    # )
    # HAI_DATASET_VALIDATION = HaiDataset(
    #     normalized_data['Valid']['timestamps'], normalized_data['VALIDATION']['dataframe'],
    #     WINDOW_GIVEN, WINDOW_SIZE, attacks=normalized_data['VALIDATION']['attacks']
    # )
    # HAI_DATASET_TEST = HaiDataset(
    #     normalized_data['Test']['timestamps'], normalized_data['TEST']['dataframe'],
    #     WINDOW_GIVEN, WINDOW_SIZE, attacks=None
    # )

    HAI_DATASET_TRAIN = check_datafile(data_path, processed_dataset_path, 'Train',
                                       WINDOW_GIVEN, WINDOW_SIZE, stride=train_stride)
    HAI_DATASET_TEST = check_datafile(data_path, processed_dataset_path, 'Test',
                                       WINDOW_GIVEN, WINDOW_SIZE, stride=1)
    HAI_DATASET_VALIDATION = check_datafile(data_path, processed_dataset_path, 'Valid',
                                       WINDOW_GIVEN, WINDOW_SIZE, stride=1)

    params = dataloader_params.copy()
    trainloader = DataLoader(HAI_DATASET_TRAIN, **params)

    params['shuffle'] = False
    validloader = DataLoader(HAI_DATASET_VALIDATION, **params)
    testloader = DataLoader(HAI_DATASET_TEST, **params)

    return trainloader, validloader, testloader