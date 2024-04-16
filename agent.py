import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
import pygame

from minesweeper import Minesweeper
from model import Deep_QNet

pygame.init()

MAX_SIZE = 1000000
BATCH = 512

LR = 0.001

EPSILON = 0.9
EPSILON_DECAY = 0.99975
EPSILON_MIN = 0.001

GAMMA = 0.1

UPDATE_TARGET_EVERY = 5


class Agent:
    def __init__(self, device, nrows, ncols):
        self.n_games = 0
        self.update_counter = 0
        self.epsilon = EPSILON
        self.lr = LR
        self.memory = deque(maxlen=MAX_SIZE)
        self.reward = 0
        self.done = False
        self.device = device
        self.model = Deep_QNet(nrows, ncols)
        self.model.to(device=device)
        self.target_model = Deep_QNet(nrows, ncols)
        self.target_model.to(device=device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def get_state(self, game):
        state = np.expand_dims(game.playerfield, axis=0)
        state = state / 8
        return state

    def get_action(self, state):
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= 0.0005
        if random.random() > self.epsilon:
            state0 = torch.tensor(state, dtype=torch.float, device=self.device)
            state0 = state0.unsqueeze(0)
            pred = self.model(state0)
            # Set pred to minimum value if the move is already revealed
            pred[0, state0.reshape(-1) != -0.125] = -1
            move = torch.argmax(pred)
        else:
            unsolved = [i for i, x in enumerate(state.reshape(-1)) if x == -0.125]
            move = torch.tensor(random.choice(unsolved))

        return move.cpu()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def train_step(self, done):
        if len(self.memory) > BATCH:
            mini_sample = random.sample(self.memory, BATCH)
        else:
            mini_sample = self.memory

        current_states = np.array([sample[0] for sample in mini_sample])
        current_qs = self.model(torch.tensor(current_states, dtype=torch.float, device=self.device))

        new_current_states = np.array([sample[3] for sample in mini_sample])
        future_qs = self.target_model(torch.tensor(new_current_states, dtype=torch.float, device=self.device))

        y = current_qs.detach().clone()

        for index, (state, action, reward, next_state, done) in enumerate(mini_sample):
            if not done:
                max_future_q = torch.max(future_qs[index])
                new_q = reward + GAMMA * max_future_q
            else:
                new_q = reward

            y[index][action] = new_q

        self.optimizer.zero_grad()
        loss = self.criterion(current_qs, y)
        loss.backward()
        self.optimizer.step()

        if done:
            self.update_counter += 1

        if self.update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.update_counter = 0

        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


def train(nrows, ncols, nmines):
    # device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    scores = deque(maxlen=1000)
    wins = deque(maxlen=1000)
    agent = Agent(device, nrows=nrows, ncols=ncols)
    game = Minesweeper(nrows=nrows, ncols=ncols, mine_count=nmines, gui=True)
    old_state = game.reset()
    highest_win_rate = 0
    action = 0
    game_reward = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        done = False
        if not done:
            action = agent.get_action(old_state)
            new_state, reward, done = game.step(action)

            agent.remember(old_state, action, reward, new_state, done)

            agent.train_step(done)

            game.render()

            old_state = new_state
            game_reward += reward

        if done:
            game.plot_minefield(action)
            agent.n_games += 1
            scores.append(game.score)
            mean_score = sum(scores) / len(scores)
            if not game.explosion:
                wins.append(1)
            else:
                wins.append(0)
            win_rate = sum(wins) / len(wins)
            print(f'Game {agent.n_games:<10}\t Score: {game.score:<10}\t'
                  f'Win Rate: {win_rate:<10.2f}\t '
                  f'Average Score: {mean_score:<10.1f}\t Epsilon: {agent.epsilon:<10.2f}\t '
                  f'Moves: {game.move_num:<10} Reward: {game_reward:.1f}')
            if win_rate >= highest_win_rate:
                agent.model.save()
            old_state = game.reset()
            game_reward = 0


if __name__ == '__main__':
    train(4, 4, 3)
