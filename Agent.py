import torch
from collections import deque
from Minesweeper import Game
import numpy as np

MAX_SIZE = 1_000_000
BATCH = 1_000
LR = 0.01


class Agent:
    def __init__(self):
        self.n_games = 0
        # self.epsilon = 0
        self.memory = deque(maxlen=MAX_SIZE)
        self.game = Game()
        # self.model
        self.reward = 0
        self.done = False
        self.score = 0

    def get_state(self, x, y):
        state = []
        grid = self.game.grid[y][x]
        state.append(grid.val)
        state.append(grid.free)
        # [0, 1, ..]
        height_width = self.game.game_width_height
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

    def get_move(self, state):
        pass

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def play_step(self, action):
        self.reward, self.done, self.score = self.game.play_step(action)

    def train(self):
        old_state = self.get_state()
