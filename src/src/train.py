import torch
import time
import datetime
import numpy as np
from models import *
from tqdm.notebook import trange
import sys
from utils.utils import EarlyStopping

class ModelTrain:
    def __init__(self,
                 model,
                 trainloader,
                 validloader,
                 epochs,
                 criterion,
                 optimizer,
                 device,
                 early_patience,
                 early_verbose,
                 model_params):
        self.model = globals()[model](**model_params)
        self.trainloader = trainloader
        self.validloader = validloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.history = {}

        self.epochs = trange(epochs, desc="Training")
        self.data_shape = next(iter(self.trainloader)).shape

        early_stopping = EarlyStopping(patience=early_patience, verbose=early_verbose)

        for e in epochs:
            self.train(e)

        CHECK_TS, CHECK_DIST, CHECK_ATT = self.validation()
        ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)


    def train(self, e):
        self.model.train()
        epoch_loss = 0
        best = {"loss": sys.float_info.max}

        for batch in self.trainloader:
            self.optimizer.zero_grad()
            given = batch["given"].cuda()
            guess = self.model(given)
            answer = batch["answer"].cuda()
            loss = self.criterion(answer, guess)
            loss.backward()
            epoch_loss += loss.item()
            self.optimizer.step()
        self.history.setdefault('loss', []).append(epoch_loss)
        self.epochs.set_postfix_str(f"loss: {epoch_loss:.6f}")
        if epoch_loss < best["loss"]:
            best["state"] = self.model.state_dict()
            best["loss"] = epoch_loss
            best["epoch"] = e + 1

    def validation(self):
        self.model.eval()
        ts, dist, att = [], [], []
        with torch.no_grad():
            for batch in self.validloader:
                given = batch["given"].cuda()
                answer = batch["answer"].cuda()
                guess = self.model(given)
                ts.append(np.array(batch["ts"]))
                dist.append(torch.abs(answer - guess).cpu().numpy())
                try:
                    att.append(np.array(batch["attack"]))
                except:
                    att.append(np.zeros(self.data_shape[0]))

        return (
            np.concatenate(ts),
            np.concatenate(dist),
            np.concatenate(att),
        )


    def model_save(self, checkpoint_path):
        with open(checkpoint_path, "wb") as f:
            torch.save(
                {
                    "state": self.best_model["state"],
                    "best_epoch": self.best_model["epoch"],
                    "loss_history": self.history['loss'],
                },
                f,
            )
