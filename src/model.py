import torch
import time
import numpy as np
from models.GRU import *
from models.GPT import *
from models.LSTMAE import *
from models.CONV1dAE import *
from models.TransformerEncoder import *
import os
import sys
from utils.utils import EarlyStopping, AverageMeter, ProgressMeter

class Model:
    def __init__(self,
                 model,
                 trainloader,
                 validloader,
                 epochs,
                 device,
                 early_patience,
                 early_verbose,
                 checkpoint_path,
                 model_params,
                 checkpoint=None):
        self.model_name = model
        self.checkpoint_path = checkpoint_path
        input_shape = next(iter(trainloader))['given'].shape[1:]
        self.model = globals()[model](input_shape[-1], **model_params).to(device)

        if checkpoint is not None:
            self.model_load(checkpoint)

        self.trainloader = trainloader
        self.validloader = validloader
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.device = device
        self.history = {}
        self.best = {"loss": sys.float_info.max}

        self.epochs = epochs
        self.early_stopping = EarlyStopping(patience=early_patience, verbose=early_verbose)


    def train(self, progress_display=False):
        self.model.train()
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            len(self.trainloader),
            [batch_time, losses],
            prefix="Epoch: [{}]".format(self.epochs))

        end = time.time()
        for e in range(1, self.epochs+1):
            for idx, batch in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                given = batch["given"].to(self.device)
                answer = batch["answer"].to(self.device)
                guess = self.model(given)
                # print(guess.shape, answer.shape)
                loss = self.criterion(answer, guess)
                loss.backward()
                self.optimizer.step()

                losses.update(loss.item(), batch["given"].size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                if progress_display == True and idx % 10 == 0:
                    progress.display(idx)

            self.history.setdefault('loss', []).append(losses.avg)

            if losses.avg < self.best["loss"]:
                self.best["state"] = self.model.state_dict()
                self.best["loss"] = losses.avg
                self.best["epoch"] = e + 1

            if e % 10 == 0:
                print(f"[Train] Epoch : {e:^3}" \
                      f"  Train Loss: {losses.avg:.4}")
                checkpoint_name = f'{self.model_name}_{e}.tar'
                self.model_save(os.path.join(self.checkpoint_path, checkpoint_name))

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
                "state": self.best["state"],
                "best_epoch": self.best["epoch"],
                "loss_history": self.history['loss'],
            },
            checkpoint
        )

    def model_load(self, checkpoint):
        self.model.load_state_dict(torch.load(checkpoint)['state'])
