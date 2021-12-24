import torch
import torch.nn as nn


class CONV1dAE(nn.Module):
    def __init__(self, input_channel, kernel_size=7, stride=2, padding=3):
        super(CONV1dAE, self).__init__()

        self.ad_layer = nn.Sequential(
            nn.Conv1d(input_channel, 32, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 16, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 16, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose1d(16, 32, kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose1d(32, input_channel, kernel_size, padding=padding),
        )

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        output = self.ad_layer(inputs)
        return output.permute(0, 2, 1)