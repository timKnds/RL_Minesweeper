import torch
import torch.nn as nn
import torch.nn.functional as F


class Deep_QNet(nn.Module):
    def __init__(self, input_channels, shape):
        super().__init__()
        # output_shape = (128, 16, 16)
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1,
                               padding_mode='zeros', bias=True)
        # output_shape = (128, 14, 14)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1,
                               padding_mode='zeros', bias=True)
        # output_shape = (64, 13, 13)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,
                               padding_mode='zeros', bias=True)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1,
                               padding_mode='zeros', bias=True)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1,
                               padding_mode='zeros', bias=True)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1,
                               padding_mode='zeros', bias=True)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc1 = nn.Linear(shape[0] * shape[1] * 32, 512)
        self.fc2 = nn.Linear(512, shape[0] * shape[1])

        # self.linear1 = nn.Linear(input_size, 4 * input_size)
        # self.linear2 = nn.Linear(4 * input_size, 4 * output_size)
        # self.linear3 = nn.Linear(4 * output_size, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        # x = x.reshape(x.shape[0], -1)
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
        else:
            x = torch.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

    def save(self, name="rl_agent.pth"):
        torch.save(self.state_dict(), name)
