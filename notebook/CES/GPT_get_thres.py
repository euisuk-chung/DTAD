import sys

from pathlib import Path
from datetime import timedelta

import dateutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import trange
from TaPR_pkg import etapr
from src import dataset
import random
import pickle
from torchinfo import summary
# import wandb

import datetime
now = datetime.datetime.now()
# wandb.run.name = now.strftime("%Y/%m/%d %H:%m:%S")

import transformers
from src.models.gpt2 import GPT2Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

WINDOW_GIVEN = 254
WINDOW_SIZE = 255

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
    
def boundary_check(df):
    x = np.array(df, dtype=np.float32)
    return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))

TRAIN_DATASET = sorted([x for x in Path("HAICon2021_dataset/train/").glob("*.csv")])
# TRAIN_DATASET

VALIDATION_DATASET = sorted([x for x in Path("HAICon2021_dataset/validation/").glob("*.csv")])
# VALIDATION_DATASET

TEST_DATASET = sorted([x for x in Path("HAICon2021_dataset/test/").glob("*.csv")])
# TEST_DATASET

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)

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

# TRAIN
TRAIN_DF = normalize(TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()
HAI_DATASET_TRAIN = HaiDataset(TRAIN_DF_RAW[TIMESTAMP_FIELD], TRAIN_DF, stride=10)

# VAL
VALIDATION_DF_RAW = dataframe_from_csvs(VALIDATION_DATASET)
VALIDATION_DF = normalize(VALIDATION_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()

HAI_DATASET_VALIDATION = HaiDataset(
    VALIDATION_DF_RAW[TIMESTAMP_FIELD], VALIDATION_DF, attacks=VALIDATION_DF_RAW[ATTACK_FIELD]
)

# TEST
TEST_DF_RAW = dataframe_from_csvs(TEST_DATASET)
TEST_DF = normalize(TEST_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()

HAI_DATASET_TEST = HaiDataset(
    TEST_DF_RAW[TIMESTAMP_FIELD], TEST_DF, attacks=None
)

params = {'batch_size': 512,
          'shuffle': True,
          'num_workers': 4,
          'pin_memory' : True}
trainloader = DataLoader(HAI_DATASET_TRAIN, **params)
params['shuffle'] = False
validloader = DataLoader(HAI_DATASET_VALIDATION, **params)
testloader = DataLoader(HAI_DATASET_TEST, **params)

train_shape = next(iter(trainloader))['given'].shape

class HAIGPT(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_size,
            max_len=WINDOW_GIVEN,
            device='cuda',
            **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.device = device
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, hidden_size))
        self.embed_token = nn.Linear(self.input_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict = nn.Linear(hidden_size, self.input_dim)

    def forward(self, token, attention_mask=None):

        batch_size, seq_length = token.shape[0], token.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(self.device)

        # embed each modality with a different head
        token_embeddings = self.embed_token(token)

        # time embeddings are treated similar to positional embeddings
        token_embeddings = token_embeddings + self.pos_embedding

        # which works nice in an autoregressive sense since states predict actions
        inputs = self.embed_ln(token_embeddings)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        
        # get predictions
        preds = self.predict(x)  

        return preds[:,-1]
    
input_shape = train_shape[1:]
model = HAIGPT(
    input_dim=input_shape[-1],
    hidden_size=128,
    max_len=WINDOW_GIVEN,
    n_layer=3,
    n_head=4,
    n_inner=4*128,
    activation_function='relu',
    n_positions=1024,
    resid_pdrop=0.1,
    attn_pdrop=0.1,
).to(device)

optimizer = optim.AdamW(model.parameters())
loss_fn = nn.MSELoss()

def train(model, train_data, optimizer, loss_fn, use_fp16=True, max_norm=None):
    
    epoch_loss = 0
    
    model.train() 

    for idx, batch in enumerate(train_data):
        
        optimizer.zero_grad(set_to_none=True)
        scaler = torch.cuda.amp.GradScaler()
                
        input = batch['given'].to(device)
        answer = batch["answer"].to(device)
        
        with torch.cuda.amp.autocast(enabled=use_fp16):
            predictions = model.forward(input)
            train_loss = loss_fn(predictions, answer)
        if use_fp16:
            scaler.scale(train_loss).backward()
            if max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.backward()
            optimizer.step()
        
        epoch_loss += train_loss.item()
        
    return epoch_loss

def validation(model, val_data, loss_fn):
    model.eval()
    val_loss = 0
    for idx, batch in enumerate(val_data):
        input = batch['given'].to(device)
        answer = batch["answer"].to(device)
        with torch.no_grad():
            predictions = model.forward(input)
            val_loss += loss_fn(predictions, answer).item()
        
    return val_loss

class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print(f'\n Training process is stopped early....')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False

def inference(dataloader, model):
    model.eval()
    ts, pred, dist, att = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            given = batch['given'].to(device)
            answer = batch["answer"].to(device)
            guess = model(given)
            ts.append(np.array(batch["ts"]))
            pred.append(guess.cpu().numpy())
            dist.append(torch.abs(answer - guess).cpu().numpy())
#             dist.append(torch.sum(torch.abs(guess-given), 1).cpu().numpy())
            try:
                att.append(np.array(batch["attack"]))
            except:
                att.append(np.zeros(given.shape[0]))
            
    return (
        np.concatenate(ts),
        np.concatenate(pred),
        np.concatenate(dist),
        np.concatenate(att),
    )

def range_check(series, size):
    size = size
    data = []

    for i in range(len(series)-size+1):
        if i == 0 :
            check_std = np.std(series[i:i+size])
        std = np.std(series[i:i+size])
        mean = np.mean(series[i:i+size])
        max = np.max(series[i:i+size])
        if check_std * 2 >= std:
            check_std = std
            data.append(mean)
        elif max == series[i]:
            data.append(max*5)
            check_std = std
        else:
            data.append(series[i]*3)
    for _ in range(size-1):
        data.append(mean)

    return np.array(data)

# LOAD MODEL
model.load_state_dict(torch.load('GPT_254.tar')['model'])

from einops import rearrange, repeat

def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs

def fill_blank(check_ts, labels, total_ts):
    def ts_generator():
        for t in total_ts:
            yield dateutil.parser.parse(t)

    def label_generator():
        for t, label in zip(check_ts, labels):
            yield dateutil.parser.parse(t), label

    g_ts = ts_generator()
    g_label = label_generator()
    final_labels = []

    try:
        current = next(g_ts)
        ts_label, label = next(g_label)
        while True:
            if current > ts_label:
                ts_label, label = next(g_label)
                continue
            elif current < ts_label:
                final_labels.append(0)
                current = next(g_ts)
                continue
            final_labels.append(label)
            current = next(g_ts)
            ts_label, label = next(g_label)
    except StopIteration:
        return np.array(final_labels, dtype=np.int8)
    
import numpy as np
from collections import Counter
import pickle as pkl

# ## VALIDATION
# model.eval()
# CHECK_TS, PRED, CHECK_DIST, CHECK_ATT = inference(validloader, model)

# ATTACK_LABELS = put_labels(np.array(VALIDATION_DF_RAW[ATTACK_FIELD]), threshold=0.5)
# # Counter(ATTACK_LABELS), ATTACK_LABELS.shape

# CHECK_DIST_DF = pd.DataFrame(CHECK_DIST)
# ab_dist = {}
# ab_thres = {}

# # GET BEST THRESHOLD
# for i in range(CHECK_DIST_DF.shape[1]):
#     print(f'Get best threshold for col {i}')
    
#     best_thres = 0
#     best_f1 = -1
    
#     #ANOMALY_SCORE =  CHECK_DIST_DF[i].to_numpy()
#     ANOMALY_SCORE =  range_check(CHECK_DIST_DF[i].to_numpy(), size=30)
    
#     ab_dist[f'ANOMALY_SCORE_{i}'] = ANOMALY_SCORE

# #     check_graph(ANOMALY_SCORE, CHECK_ATT, piece=2, THRESHOLD=THRESHOLD, title = True, title_nm = i)
    
#     # get stats
#     MIN_A_SCORE = min(ANOMALY_SCORE)
#     MAX_A_SCORE = max(ANOMALY_SCORE)
#     A_SCORE_DIFF = MAX_A_SCORE - MIN_A_SCORE
    
#     # search params
#     for THRESHOLD in list(np.around(np.arange(MIN_A_SCORE, MAX_A_SCORE, A_SCORE_DIFF/50), 6)):
#         print(f'----------THRESHOLD : {THRESHOLD}----------')
#         LABELS = put_labels(ANOMALY_SCORE, THRESHOLD)
#         FINAL_LABELS = fill_blank(CHECK_TS, LABELS, np.array(VALIDATION_DF_RAW[TIMESTAMP_FIELD]))
        
#         print('>>> ATTACK_LABELS')
#         print(Counter(ATTACK_LABELS), ATTACK_LABELS.shape)
#         print('>>> FINAL_LABELS')
#         print(Counter(FINAL_LABELS), FINAL_LABELS.shape)

#         if ATTACK_LABELS.shape[0] == FINAL_LABELS.shape[0]:
#             TaPR = etapr.evaluate_haicon(anomalies=ATTACK_LABELS, predictions=FINAL_LABELS)
#             print(f">>> F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
#             print(f">>> # of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
#             print(f">>> Detected anomalies: {TaPR['Detected_Anomalies']}")

#             if TaPR['f1']>best_f1:
#                 best_thres = THRESHOLD
#                 best_f1 = TaPR['f1']
#                 print("***Best Threshold updated!!***")
                    
#     print(f'>>>> BEST THRESHOLD FOR COL{i} is {best_thres} and F1 is {best_f1}!!!')
    
#     ab_thres[f'ANOMALY_THRES_{i}'] =  best_thres
    
#     # 만일 MIN이라면 사용 X
#     if best_thres == MIN_A_SCORE :
#         print(f'Not using col {i} !!')
#         ab_thres[f'ANOMALY_THRES_{i}'] = None

#     # SAVE
#     with open('./ANOMALY_SCORE.pkl', 'wb') as A_SCORE:
#         pkl.dump(ab_dist, A_SCORE)
#     with open('./ANOMALY_THRES.pkl', 'wb') as A_THRES:
#         pkl.dump(ab_thres, A_THRES)
        
# #     with open('filename.pkl', 'rb') as handle:
# #         b = pickle.load(handle)

# LOAD RESULTS
with open('./ANOMALY_SCORE.pkl', 'rb') as A_SCORE:
    ab_dist = pkl.load(A_SCORE)

with open('./ANOMALY_THRES.pkl', 'rb') as A_THRES:
    ab_thres = pkl.load(A_THRES)


# TEST
model.eval()
CHECK_TS, PRED, CHECK_DIST, CHECK_ATT = inference(testloader, model)

CHECK_DIST_DF = pd.DataFrame(CHECK_DIST)
final_pred = {}

# GET BEST THRESHOLD
for i in range(CHECK_DIST_DF.shape[1]):
    print(f'Using best threshold for col {i}')
    
    if ab_thres[f'ANOMALY_THRES_{i}'] != None:
        
        ANOMALY_SCORE =  range_check(CHECK_DIST_DF[i].to_numpy(), size=30)
        
        THRESHOLD = ab_thres[f'ANOMALY_THRES_{i}']
        
        LABELS = put_labels(ANOMALY_SCORE, THRESHOLD)
        
        FINAL_LABELS = fill_blank(CHECK_TS, LABELS, np.array(TEST_DF_RAW[TIMESTAMP_FIELD]))
        
        final_pred[f'COL{i}'] = FINAL_LABELS
    
    else:
        pass
    
    # SAVE
    with open('./FINAL_PRED.pkl', 'wb') as F_PRED:
        pkl.dump(final_pred, F_PRED)
        
