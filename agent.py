import torch
import random
import numpy as np
from collections import deque
import pygame
import torch.multiprocessing as mps

from minesweeper import Minesweeper
from model import Linear_QNet, QTrainer
from helper import plot

pygame.init()

MAX_SIZE = 1_000_000
BATCH = 1_000
LR = 0.001
EPSILON_MIN = 0.01

class Agent:
    def __init__(self, model, device):
        self.n_games = 0
        self.epsilon = 0.99
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_SIZE)
        self.reward = 0
        self.done = False
        # self.x_model = Linear_QNet(22,  20)
        # self.y_model = Linear_QNet(22, 20)
        self.device = device
        self.model = model
        self.model.to(device=device)
        # self.model.share_memory()
        self.trainer = QTrainer(LR, self.gamma, self.model, self.device)

    def get_action(self, state, n_games):
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= 0.0005
        final_move = torch.tensor(random.randint(0, 255))
        if random.random() > self.epsilon:
            state0 = torch.tensor(state, dtype=torch.float, device=self.device)
            # state0 = torch.unsqueeze(state0, 0)
            final_move = self.model(state0)
            final_move = torch.round(final_move.cpu(), decimals=0)
            final_move = torch.tensor(final_move.item(), dtype=torch.int32)
            final_move = final_move.item()
            if final_move > 255:
                final_move = 255

        return final_move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def train_long_memory(self):
        if len(self.memory) > BATCH:
            mini_sample = random.sample(self.memory, BATCH)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


def train(model, shape):
    # device agnostic code
    if not torch.backends.mps.is_available():
        mps_device = torch.device("cpu")
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")

    else:
        mps_device = torch.device("mps")

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    reward = 0
    agent = Agent(model, mps_device)
    game = Minesweeper(shape[0], shape[1], mine_count=shape[2], gui=True)
    old_state = game.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        done = False
        if not done:
            action = agent.get_action(old_state, agent.n_games)
            new_state, reward, done = game.step(action)
            game.render()
            # game.timer.tick(15)
            agent.train_short_memory(old_state, action, reward, new_state, done)

            agent.remember(old_state, action, reward, new_state, done)

        if done:
            game.plot_minefield(action)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            print(f'Game {agent.n_games}, Score: {game.score}, Record: {record}')
            if game.score > record:
                record = game.score
                agent.model.save()
            plot_scores.append(game.score)
            total_score += game.score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    shape = (16, 16, 20)

    model = Linear_QNet(1, 1, shape)
    model.share_memory()

    train(model, shape)
    # num_processes = 3
    # processes = []
    # for rank in range(num_processes):
    #     p = mps.Process(target=train, args=(model, shape))
    #     print(f"Process {rank} started")
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
