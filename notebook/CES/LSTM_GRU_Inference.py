import sys

from pathlib import Path
from datetime import timedelta

import dateutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import trange
from TaPR_pkg import etapr

from collections import Counter
import random

# GPU 할당 변경하기
GPU_NUM = 1 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

torch.cuda.set_device(device) # change allocation of current GPU
torch.backends.cudnn.benchmark = True

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(72)

TRAIN_DATASET = sorted([x for x in Path("./HAICon2021_dataset/train/").glob("*.csv")])
# TRAIN_DATASET

VALIDATION_DATASET = sorted([x for x in Path("./HAICon2021_dataset/validation/").glob("*.csv")])
# VALIDATION_DATASET

TEST_DATASET = sorted([x for x in Path("./HAICon2021_dataset/test/").glob("*.csv")])
# TEST_DATASET

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)
TRAIN_DF_RAW

TIMESTAMP_FIELD = "timestamp"
IDSTAMP_FIELD = 'id'
ATTACK_FIELD = "attack"
VALID_COLUMNS_IN_TRAIN_DATASET = TRAIN_DF_RAW.columns.drop([TIMESTAMP_FIELD])
VALID_COLUMNS_IN_TRAIN_DATASET

TAG_MIN = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].min()
TAG_MAX = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].max()

def normalize(df):
    ndf = df.copy()
    for c in df.columns:
        if TAG_MIN[c] == TAG_MAX[c]:
            ndf[c] = df[c] - TAG_MIN[c]
        else:
            ndf[c] = (df[c] - TAG_MIN[c]) / (TAG_MAX[c] - TAG_MIN[c])
    return ndf

TRAIN_DF = normalize(TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()
#TRAIN_DF

WINDOW_GIVEN = 89
WINDOW_SIZE = 90

class HaiDataset(Dataset):
    def __init__(self, timestamps, df, stride=1, attacks=None):
        self.ts = np.array(timestamps)
        self.tag_values = np.array(df, dtype=np.float32)
        self.valid_idxs = []
        for L in trange(len(self.ts) - WINDOW_SIZE + 1):
            R = L + WINDOW_SIZE - 1
            if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(
                self.ts[L]
            ) == timedelta(seconds=WINDOW_SIZE - 1):
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
        last = i + WINDOW_SIZE - 1
        item = {"attack": self.attacks[last]} if self.with_attack else {}
        item["ts"] = self.ts[i + WINDOW_SIZE - 1]
        item["given"] = torch.from_numpy(self.tag_values[i : i + WINDOW_GIVEN])
        item["answer"] = torch.from_numpy(self.tag_values[last])
        return item
    
HAI_DATASET_TRAIN = HaiDataset(TRAIN_DF_RAW[TIMESTAMP_FIELD], TRAIN_DF, stride=1)

N_HIDDENS = 128
N_LAYERS = 3
BATCH_SIZE = 512

class StackedGRU(torch.nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=n_tags,
            hidden_size=N_HIDDENS,
            num_layers=N_LAYERS,
            bidirectional=True,
            dropout=0,
        )
        self.fc = torch.nn.Linear(N_HIDDENS * 2, n_tags)

    def forward(self, x):
        x = x.transpose(0, 1)  # (batch, seq, params) -> (seq, batch, params)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        out = self.fc(outs[-1])
        return x[0] + out
    
MODEL = StackedGRU(n_tags=TRAIN_DF.shape[1])
MODEL.cuda()

best_epoch = 398

with open(f"GRU_model_{best_epoch}.pt", "rb") as f:
    SAVED_MODEL = torch.load(f)

MODEL.load_state_dict(SAVED_MODEL["state"])

VALIDATION_DF_RAW = dataframe_from_csvs(VALIDATION_DATASET)
VALIDATION_DF = normalize(VALIDATION_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()

ATTACK_LABELS = VALIDATION_DF_RAW['attack']

VALID_INFERENCE = VALIDATION_DF.copy()
VALID_INFERENCE=VALID_INFERENCE.reset_index(drop=True)
VALID_INFERENCE.head()

VAL_LEN = VALID_INFERENCE.shape[0]

best_thres = 0
best_f1 = 0

# 1000개 param search
for THRESHOLD in list(np.around(np.arange(0.010, 0.020, 0.00001),5)):
    print(f'----------THRESHOLD : {THRESHOLD}----------')
    val_att = []

    for i in range(0, VAL_LEN - WINDOW_GIVEN):
        #초기화
        attack = 0

        # model loads
        given = np.array(VALID_INFERENCE.iloc[i:i+WINDOW_GIVEN])
        answer = np.array(VALID_INFERENCE.iloc[i+WINDOW_GIVEN])

        # match shape
        given = torch.from_numpy(given).float().unsqueeze(0).to(device)
        answer = torch.from_numpy(answer).unsqueeze(0).to(device)

        guess = MODEL(given) 

        # diff
        diff = torch.abs(answer - guess)
        diff = np.mean(diff.cpu().detach().numpy())
        #print(f'i : {i} & diff : {diff}')

        if diff >= THRESHOLD:
            attack = 1
            VALID_INFERENCE.iloc[i+WINDOW_GIVEN] = np.array(guess.squeeze(0).cpu().detach().numpy())

        val_att.append(attack)

    # fill blanks
    add_zeros = [0]*WINDOW_GIVEN
    FINAL_LABELS = np.concatenate((add_zeros, val_att), axis=None)
    
    # print
    print(f'>>>> ATTACK_LABELS : {Counter(ATTACK_LABELS)}')
    print(f'>>>> FINAL_LABELS : {Counter(FINAL_LABELS)}')
    
    #  check
    if ATTACK_LABELS.shape[0] == FINAL_LABELS.shape[0]:
        TaPR = etapr.evaluate_haicon(anomalies=ATTACK_LABELS, predictions=FINAL_LABELS)
        print(f">>> F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
        print(f">>> # of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
        print(f">>> Detected anomalies: {TaPR['Detected_Anomalies']}")
        
        if TaPR['f1']>best_f1:
            
            print("***Best Threshold updated!!***")
            
            best_thres = THRESHOLD
            best_f1 = TaPR['f1']
            
            # save
            save_list = [best_thres, best_f1]
            with open('GRU_val_best_thres_f1.pkl', 'wb') as f:
                pickle.dump(save_list, f)
            print('save complete!')
            

#             with open('GPT_val_best_thres_f1.pkl', 'rb') as f:
#                 best_thres = pickle.load(f)

# -----------------------------------------------------------

# TEST
TEST_INFERENCE = TEST_DF.copy()
TEST_INFERENCE=TEST_INFERENCE.reset_index(drop=True)
# TEST_INFERENCE.head()

WINDOW_GIVEN = 89
WINDOW_SIZE = 90
THRESHOLD = best_thres

TEST_LEN = TEST_INFERENCE.shape[0]
att = []

for i in trange(0, TEST_LEN - WINDOW_GIVEN):
    #초기화
    attack = 0

    # model loads
    given = np.array(TEST_INFERENCE.iloc[i:i+WINDOW_GIVEN])
    answer = np.array(TEST_INFERENCE.iloc[i+WINDOW_GIVEN])
    
    # match shape
    given = torch.from_numpy(given).float().unsqueeze(0).to(device)
    answer = torch.from_numpy(answer).unsqueeze(0).to(device)
    
    guess = MODEL(given) 

    # diff
    diff = torch.abs(answer - guess)
    diff = np.mean(diff.cpu().detach().numpy())
    #print(f'i : {i} & diff : {diff}')

    if diff >= THRESHOLD:
        attack = 1
        TEST_INFERENCE.iloc[i+WINDOW_GIVEN] = np.array(guess.squeeze(0).cpu().detach().numpy())
    
    att.append(attack)

add_zeros = [0]*WINDOW_GIVEN
# print(len(add_zeros))

concat_attack = np.concatenate((add_zeros, att), axis=None)
# print(len(concat_attack))
# print()
# print(concat_attack)

submission = pd.read_csv('./HAICon2021_dataset/sample_submission.csv')
# submission.shape

submission['attack'] = pd.Series(concat_attack)
# submission

submission.to_csv(f'GRU_89_renewal_{THRESHOLD}.csv', index=False)