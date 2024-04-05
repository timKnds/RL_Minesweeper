import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
import pygame
# import torch.multiprocessing as mps

from minesweeper import Minesweeper
from model import Deep_QNet

pygame.init()

MAX_SIZE = 1_000_000
BATCH = 1_000

LR = 0.01
LR_DECAY = 0.99975
LR_MIN = 0.0001

EPSILON = 0.9
EPSILON_DECAY = 0.99975
EPSILON_MIN = 0.001

GAMMA = 0.9

UPDATE_TARGET_EVERY = 5


class Agent:
    def __init__(self, device, shape):
        self.n_games = 0
        self.update_counter = 0
        self.epsilon = EPSILON
        self.lr = LR
        self.memory = deque(maxlen=MAX_SIZE)
        self.reward = 0
        self.done = False
        self.device = device
        self.model = Deep_QNet(1, shape)
        self.model.to(device=device)
        self.target_model = Deep_QNet(1, shape)
        self.target_model.to(device=device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        # self.model.share_memory()

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
            pred[0, state0.reshape(-1) != -0.125] = torch.min(pred)
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

        X, y = [], []

        for index, (state, action, reward, next_state, done) in enumerate(mini_sample):
            if not done:
                max_future_q = torch.max(future_qs[index])
                new_q = reward + GAMMA * max_future_q
            else:
                new_q = reward

            current_q = current_qs[index]
            current_q[action] = new_q

            X.append(state.reshape(-1))
            y.append(current_q.detach().cpu())

        self.optimizer.zero_grad()
        loss = self.criterion(np.array(X, dtype=np.float32), np.array(y, dtype=np.float32))
        loss.backward()
        self.optimizer.step()

        if done:
            self.update_counter += 1

        if self.update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.update_counter = 0

        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.lr = max(LR, self.lr * LR_DECAY)


def train(shape):
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

    scores = deque(maxlen=100)
    wins = deque(maxlen=100)
    record = 0
    agent = Agent(mps_device, shape)
    game = Minesweeper(shape[0], shape[1], mine_count=shape[2], gui=True)
    old_state = game.reset()
    action = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        done = False
        game_reward = 0
        if not done:
            action = agent.get_action(old_state)
            new_state, reward, done = game.step(action)
            game.render()

            agent.remember(old_state, action, reward, new_state, done)

            agent.train_step(done)

            old_state = new_state
            game_reward += reward

        if done:
            agent.n_games += 1
            scores.append(game.score)
            mean_score = sum(scores) / len(scores)
            if not game.explosion:
                wins.append(1)
            else:
                wins.append(0)
            win_rate = sum(wins) / len(wins)
            print(f'Game {agent.n_games}\t Score: {game.score}\t Record: {record}\t Win Rate: {win_rate:.3g}\t '
                  f'Average Score: {mean_score:.3g}\t Epsilon: {agent.epsilon:.3g}\t '
                  f'Moves: {game.move_num}, Reward: {game_reward}')
            game.plot_minefield(action)
            if game.score >= record:
                record = game.score
                agent.model.save()
            old_state = game.reset()


if __name__ == '__main__':
    shape = (4, 4, 3)

    # model.load_state_dict(torch.load("rl_agent.pth"))
    # model.share_memory()

    train(shape)
    # num_processes = 3
    # processes = []
    # for rank in range(num_processes):
    #     p = mps.Process(target=train, args=(model, shape))
    #     print(f"Process {rank} started")
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
