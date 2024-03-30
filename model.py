import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    def __init__(self, input_channels, output_channels, shape):
        super().__init__()
        # output_shape = (128, 6, 6)
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=0, bias=False)
        # add batch normalization
        self.batch_norm1 = nn.BatchNorm2d(128)
        # output_shape = (128, 4, 4)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(128)
        # output_shape = (64, 2, 2)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear((shape[0] - 6) * (shape[1] - 6) * 64, 512)
        self.fc2 = nn.Linear(512, output_channels)

        # self.linear1 = nn.Linear(input_size, 4 * input_size)
        # self.linear2 = nn.Linear(4 * input_size, 4 * output_size)
        # self.linear3 = nn.Linear(4 * output_size, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = x.reshape(x.shape[0], -1)
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
        else:
            x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)

        if len(state.shape) == 3:
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
        # else:
        #     state = state.squeeze()
        #     next_state = next_state.squeeze()
        #     action = action.squeeze()

        # predict th
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx]))

            target[idx] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
