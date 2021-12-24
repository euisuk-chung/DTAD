# Time Series Anomaly Detection

Github Repo for timeseries anomaly detection (시계열 이상치 탐지)

## 1. Purpose

Time Series anomaly detection is used for the purpose of discriminating abnormally generated data on time series data.

## 2. How to use

- You can check detail about the argument at `4. Model Parameters`

### 2.1. LSTMAE

```python
# Example
## Training
python main.py -M LSTMAE -T Train -G 0 -R 30 -H 0.025 -C ./checkpoint/LSTMAE_30.tar

## Test
python main.py -M LSTMAE -T Test -G 0 -R 30 -H 0.025 -C ./checkpoint/LSTMAE_30.tar
```

### 2.2. 1DCONVAE

```python
# Example
python main.py -M CONV1dAE -T Train -G 0 -R 30 -H 0.025 -C ./checkpoint/CONV1dAE_30.tar

## Test
python main.py -M CONV1dAE -T Test -G 0 -R 30 -H 0.025 -C ./checkpoint/CONV1dAE_30.tar

```

### 2.3. GRU

```python
# Example
python main.py -M GRU -T Train -G 0 -R 30 -H 0.025 -C ./checkpoint/GRU_30.tar

## Test
python main.py -M GRU -T Test -G 0 -R 30 -H 0.025 -C ./checkpoint/GRU_30.tar

```

### 2.4. GPT

```python
# Example
python main.py -M GPT -T Train -G 0 -R 30 -H 0.025 -C ./checkpoint/GPT_30.tar

## Test
python main.py -M GPT -T Test -G 0 -R 30 -H 0.025 -C ./checkpoint/GPT_30.tar

```

### 2.5. TransformerEncoder

```python
# Example
python main.py -M TransformerEncoder -T Train -G 0 -R 30 -H 0.025 -C ./checkpoint/TransformerEncoder_30.tar

## Test
python main.py -M TransformerEncoder -T Test -G 0 -R 30 -H 0.025 -C ./checkpoint/TransformerEncoder_30.tar

```

## 3. Models Used

### 3.1. LSTM Auto Encoder

<img src = 'https://github.com/euisuk-chung/timeseries-generation/blob/main/image/TimeGAN_architecture.PNG?raw=true' width="800" height="400">

- pyTorch implementation for `TimeGAN`
- Code Reference : https://github.com/d9n13lt4n/timegan-pytorch

### 3.2. 1D Convolution Auto Encoder

<img src = 'https://github.com/euisuk-chung/timeseries-generation/blob/main/image/LSTM_VAE_architecture.png?raw=true' width="800" height="400">

- pyTorch implementation for `VRAE`
- Code Reference : https://github.com/tejaslodaya/timeseries-clustering-vae

### 3.3. GRU

<img src = 'https://github.com/euisuk-chung/timeseries-generation/blob/main/image/LSTM_VAE_architecture.png?raw=true' width="800" height="400">

- pyTorch implementation for `VRAE`
- Code Reference : https://github.com/tejaslodaya/timeseries-clustering-vae

### 3.4. GPT

<img src = 'https://github.com/euisuk-chung/timeseries-generation/blob/main/image/LSTM_VAE_architecture.png?raw=true' width="800" height="400">

- pyTorch implementation for `VRAE`
- Code Reference : https://github.com/tejaslodaya/timeseries-clustering-vae

## 4. Model Arguments

### 4.1. config.yaml

```
path: 메인 path
data_path: 사용할 데이터 path
processed_dataset_path: 전처리 및 dataloader 저장 path
checkpoint_path: model checkpoint 저장 path
save_path: 시각화 저장 path
workers: 사용할 cpu 프로세서 수
seed: 랜덤 시드

# data parameters
window_given: 입력으로 넣을 시계열 데이터 window 크기
window_size: 입력과 출력을 포함한 시계열 데이터 window 크기
stride: 스트라이드 크기
loader_params:
  batch_size: 배치 사이즈
  shuffle: 셔플 유무
  num_workers: dataloader 사용시 사용할 cpu 프로세서 수
  pin_memory: 핀 메모리 사용 여부

# Train parameters
epochs: 학습 에포크 수
early_patience: early stopping 사용 시 참고 에포크 수
early_verbose: early stopping 사용 시 중단에 대한 출력 여부

# model parameters
GRU:
  n_hidden: GRU 히든 유닛 수
  n_layers: GRU 레이어 수


LSTMAE:
  seq_len: 시퀀스 길이


CONV1dAE:
  kernel_size: 커널 사이즈
  stride: 스트라이드 크기
  padding: 패딩 사이즈


TransformerEncoder:
  num_heads: multi head attention을 위한 head의 수
  seq_len: 시퀀스 길이
  ff_dim: feed-forward layers 차원
  num_transformer_blocks: transformer 레이어 수
  mlp_units: mlp layer의 유닛 수 (리스트)
  mlp_dropout: mlp layer의 dropout
  dropout: transformer encoder의 dropout

GPT:
  hidden_size: 히든 유닛 수
  max_len: 시퀀스 최대 길이
  n_layer: transformer 레이어 수
  n_head: multi head attention을 위한 head의 수
  n_inner: inner feed-forward layers 차원
  activation_function: 활성화 함수
  n_positions: position embedding 크기
  resid_pdrop: residual dropout 비율
  attn_pdrop: attention dropout 비율
```

### 4.2. Arguments

```
-M or --model `LSTMAE` or `1DCONVAE` or 'GRU' or 'GPT'# model 종류 선택

-T or --type 'Train' or 'Test' # 모델 학습 또는 테스트

-G or --gpu 0 # 사용할 GPU 번호 지정

-R or --range_check 30 # 후처리시 사용할 window smoothing 범위 지정

-H or --threshold 0.025 # anomaly detection 시 사용할 threshold 지정

-C or --checkpoint ./checkpoints/GPT_30.tar # pretrained model state 불러와서 사용 시 checkpoint path 지정

```


## Repository Structure

```
├── src
│   ├── models
│   │   ├── CONV1dAE.py
│   │   ├── GPT.py
│   │   ├── GRU.py
│   │   ├── LSTMAE.py
│   │   └── TransformerEncoder.py
│   ├── log
│   │   └── Where logs are saved
│   ├── processed_dataset
│   │   └── Where preprocessed data are saved
│   ├── checkpoints
│   │   └── Where checkpoint data are saved
│   ├── save
│   │   └── Where result plot are saved
│   ├── utils
│   │   ├── plotting.py
│   │   ├── progress.py
│   │   └── util.py
│   ├── config.py
│   ├── config.yaml
│   ├── dataset.py
│   ├── main.py
└── └── run.sh
```
