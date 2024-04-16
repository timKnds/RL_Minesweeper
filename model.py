import torch
import torch.nn as nn
import torch.nn.functional as F


class Deep_QNet(nn.Module):
    def __init__(self, nrows, ncols, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)  # (batch, 32, +-0, +-0)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)  # (batch, 128, +-0, +-0)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)  # (batch, 128, +-0, +-0)
        self.conv4 = nn.Conv2d(128, 256, 5, padding=2)  # (batch, 64, +-0, +-0)

        self.dropout1 = nn.Dropout2d(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(256 * (ncols//2) * (nrows//2), 128)
        self.fc2 = nn.Linear(128, nrows * ncols)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

    def save(self, name="rl_agent.pth"):
        torch.save(self.state_dict(), name)
