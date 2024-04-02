import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    def __init__(self, input_channels, output_channels, shape):
        super().__init__()
        # output_shape = (128, 16, 16)
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=5, stride=1, padding=2,
                               padding_mode='replicate', bias=False)
        # output_shape = (128, 14, 14)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2,
                               padding_mode='replicate', bias=False)
        # output_shape = (64, 13, 13)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2,
                               padding_mode='replicate', bias=False)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2,
                               padding_mode='replicate', bias=False)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2,
                               padding_mode='replicate', bias=False)
        self.fc1 = nn.Linear(shape[0] * shape[1] * 64, 512)
        self.fc2 = nn.Linear(512, output_channels)

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
        # x = x.reshape(x.shape[0], -1)
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
        else:
            x = torch.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def save(self, name="rl_agent.pth"):
        torch.save(self.state_dict(), name)


class QTrainer:
    def __init__(self, learning_rate, gamma, model, device):
        self.lr = learning_rate
        self.gamma = gamma
        self.model = model
        # self.x_model = x_model
        # self.y_model = y_model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.device = device

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        # action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)

        if len(state.shape) == 3:
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)

        # predict th
        current_qs_list = self.model(state)
        future_qs_list = self.model(next_state)

        target = current_qs_list.clone()
        for idx in range(len(done)):
            if not done[idx]:
                max_future_q = torch.max(future_qs_list[idx])
                Q_new = reward[idx] + self.gamma * max_future_q
            else:
                Q_new = reward[idx]

            target[idx][action[idx]] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, current_qs_list)
        loss.backward()

        self.optimizer.step()
