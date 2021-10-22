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
from collections import Counter

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

# define model
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

# TRAIN
TRAIN_DATASET = sorted([x for x in Path("HAICon2021_dataset/train/").glob("*.csv")])
#TRAIN_DATASET
    
# VALIDATION
VALIDATION_DATASET = sorted([x for x in Path("HAICon2021_dataset/validation/").glob("*.csv")])
#VALIDATION_DATASET

# TEST
TEST_DATASET = sorted([x for x in Path("HAICon2021_dataset/test/").glob("*.csv")])
#TEST_DATASET


def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])


# TRAIN
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

TRAIN_DF = normalize(TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()
# TRAIN_DF

HAI_DATASET_TRAIN = HaiDataset(TRAIN_DF_RAW[TIMESTAMP_FIELD], TRAIN_DF, stride=10)

# VALIDATION
VALIDATION_DF_RAW = dataframe_from_csvs(VALIDATION_DATASET)
VALIDATION_DF = normalize(VALIDATION_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()


# TEST
TEST_DF_RAW = dataframe_from_csvs(TEST_DATASET)
TEST_DF = normalize(TEST_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()

# trainloader
params = {'batch_size': 512,
          'shuffle': True,
          'num_workers': 4,
          'pin_memory' : True}

trainloader = DataLoader(HAI_DATASET_TRAIN, **params)
train_shape = next(iter(trainloader))['given'].shape
print(train_shape)

# Model Define
class HAIGPT(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_size,
            max_len=254,
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
    
# params
input_shape = train_shape[1:]
model = HAIGPT(
    input_dim=input_shape[-1],
    hidden_size=128,
    max_len=254,
    n_layer=3,
    n_head=1,
    n_inner=4*128,
    activation_function='relu',
    n_positions=1024,
    resid_pdrop=0.1,
    attn_pdrop=0.1,
).to(device)

optimizer = optim.AdamW(model.parameters())
loss_fn = nn.MSELoss()

# not used
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
    
# load chkpoint
model.load_state_dict(torch.load('GPT_254.tar')['model'])


# -----------------------------------------------------------

# START Validating
ATTACK_LABELS = VALIDATION_DF_RAW['attack']
#Counter(ATTACK_LABELS)

VALID_INFERENCE = VALIDATION_DF.copy()
VALID_INFERENCE=VALID_INFERENCE.reset_index(drop=True)
# VALID_INFERENCE.head()

WINDOW_GIVEN = 254
WINDOW_SIZE = 255

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

        guess = model(given) 

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
            with open('GPT_val_best_thres_f1.pkl', 'wb') as f:
                pickle.dump(save_list, f)
            print('save complete!')
            

#             with open('GPT_val_best_thres_f1.pkl', 'rb') as f:
#                 best_thres = pickle.load(f)

# -----------------------------------------------------------

# TEST
TEST_INFERENCE = TEST_DF.copy()
TEST_INFERENCE=TEST_INFERENCE.reset_index(drop=True)
# TEST_INFERENCE.head()

WINDOW_GIVEN = 254
WINDOW_SIZE = 255
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
    
    guess = model(given) 

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

submission.to_csv(f'1019_GPT_254_renewal_{THRESHOLD}.csv', index=False)