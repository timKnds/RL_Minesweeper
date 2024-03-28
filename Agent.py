import torch
import random
import numpy as np
from collections import deque

from Minesweeper import Game
from Model import Linear_QNet, QTrainer

MAX_SIZE = 1_000_000
BATCH = 1_000
LR = 0.01


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_SIZE)
        self.reward = 0
        self.done = False
        self.model = Linear_QNet(6,  4)
        self.trainer = QTrainer(LR, self.gamma, self.model)

    def get_state(self, game, x, y):
        grid = game.grid[y][x]
        if grid.clicked:
            state = [grid.val, grid.free]
            # [0, 1, ..]
            height_width = game.game_width_height
            if x == height_width - 1:
                state.extend((0, 1))
                # [0, 1, 0, 1]
                if y == height_width - 1:
                    state.extend((0, 1))
                # [0, 1, 1, 0]
                if y == 0:
                    state.extend((1, 0))
                else:
                    state.extend((0, 0))
            # [1, 0, ..]
            if x == 0:
                state.extend((1, 0))
                # [1, 0, 0, 1]
                if y == height_width - 1:
                    state.extend((0, 1))
                # [1, 0, 1, 0]
                if y == 0:
                    state.extend((1, 0))
                else:
                    state.extend((0, 0))

            return state
        else:
            return None

    def get_action(self, state):
        pass

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH:
            mini_sample = random.sample(self.memory, BATCH)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # self.trainer.train_step(state, action, reward, next_state, done)
        pass


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game()
    done = False
    game.reset()
    while True:
        score = 0
        for x in range(game.game_width_height):
            for y in range(game.game_width_height):
                old_state = agent.get_state(game, x, y)
                if old_state and not done:
                    action = agent.get_action(old_state)
                    reward, done, score = game.play_step(action)
                    new_state = agent.get_state(game, x, y)

                    agent.train_short_memory(old_state, action, reward, new_state, done)

                    agent.remember(old_state, action, reward, new_state, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            print(f'Game {agent.n_games}, Score: {score}, Record: {record}')
        if score > record:
            record = score
            agent.model.save()
        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        plot_mean_scores.append(mean_score)
        # plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
