import torch
import time
import datetime
import numpy as np
from models import *

class ModelTrain:
    def __init__(self, model):
        self.model = globals()[model]()


    def train(self):


    def validation(self):
