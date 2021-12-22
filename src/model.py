import torch
import time
import datetime
import numpy as np
from models import *
from tqdm.notebook import trange
import os
import sys
from utils.utils import EarlyStopping, range_check

class Model:
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
        self.model_name = model
        self.checkpoint_path = model_params.checkpoint_path
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


    def train(self):
        self.model.train()
        epoch_loss = 0
        best = {"loss": sys.float_info.max}

        for e in self.epochs:
            for batch in self.trainloader:
                self.optimizer.zero_grad()
                given = batch["given"].to(self.device)
                guess = self.model(given)
                answer = batch["answer"].to(self.device)
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

            if e % 10 == 9:
                checkpoint_name = f'{self.model_name}_{e}.tar'
                self.model_save(os.path.join(checkpoint_path, checkpoint_name))

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


    def model_save(self, checkpoint):

        torch.save(
            {
                "state": self.best_model["state"],
                "best_epoch": self.best_model["epoch"],
                "loss_history": self.history['loss'],
            },
            checkpoint
        )

    def model_load(self, checkpoint):
        model.load_state_dict(torch.load(checkpoint)['model'])
