#!/bin/bash

## Training

python main.py -M GRU -T Train -G 0

python main.py -M GPT -T Train -G 0

python main.py -M LSTMAE -T Train -G 0

python main.py -M CONV1dAE -T Train -G 0

## Test

python main.py -M GRU -T Test -G 0

python main.py -M GPT -T Test -G 0

python main.py -M LSTMAE -T Test -G 0

python main.py -M CONV1dAE -T Test -G 0