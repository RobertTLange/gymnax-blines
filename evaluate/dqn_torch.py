import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


# Adopted from https://github.com/kenjyoung/MinAtar/blob/master/examples/dqn.py
class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):

        super(QNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(
            in_features=num_linear_units, out_features=128
        )

        self.output = nn.Linear(in_features=128, out_features=num_actions)

    def forward(self, x):
        x = get_state(np.array(x))
        x = f.relu(self.conv(x))
        x = f.relu(self.fc_hidden(x.view(x.size(0), -1)))
        return self.output(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_state(s):
    return (
        (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()
    )
