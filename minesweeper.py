#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygame
import numpy as np

# Import files
spr_emptyGrid = pygame.image.load("Sprites/empty.png")
spr_flag = pygame.image.load("Sprites/flag.png")
spr_grid = pygame.image.load("Sprites/Grid.png")
spr_grid1 = pygame.image.load("Sprites/grid1.png")
spr_grid2 = pygame.image.load("Sprites/grid2.png")
spr_grid3 = pygame.image.load("Sprites/grid3.png")
spr_grid4 = pygame.image.load("Sprites/grid4.png")
spr_grid5 = pygame.image.load("Sprites/grid5.png")
spr_grid6 = pygame.image.load("Sprites/grid6.png")
spr_grid7 = pygame.image.load("Sprites/grid7.png")
spr_grid8 = pygame.image.load("Sprites/grid8.png")
spr_mine = pygame.image.load("Sprites/mine.png")
spr_mineClicked = pygame.image.load("Sprites/mineClicked.png")
spr_mineFalse = pygame.image.load("Sprites/mineFalse.png")


class Minesweeper:

    def __init__(self, nrows, ncols, mine_count, gui=False):
        self.reward = {'win': 1.0, 'lose': -1.0, 'progress': 0.3, 'guess': -0.3, 'no_progress': -0.3}
        self.nrows = nrows  # number of tiles along the row dimension
        self.ncols = ncols  # number of tiles along the column dimension
        self.mine_count = mine_count
        self.minefield = np.zeros((nrows, ncols), dtype='int')  # The complete game state
        self.playerfield = np.ones((nrows, ncols), dtype='int') * -1  # The state the player sees
        self.explosion = False  # True if player selects mine
        self.done = False  # Game complete (win or loss)
        self.score = 0
        self.np_random = np.random.RandomState()  # For seeding the environment
        self.move_num = 0  # Track number of player moves per game
        if gui:
            self.init_gui()  # Pygame related parameters

    def step(self, action):
        # Function accepts the player's action as an input and returns the new
        # environment state, a reward, and whether the episode has terminated
        idx_x, idx_y = np.unravel_index(action, (self.nrows, self.ncols))
        if self.move_num == 0:
            self.generate_field(action)
        if self.playerfield[idx_x, idx_y] == self.minefield[idx_x, idx_y]:
            # Tile has already been revealed
            reward = self.reward['no_progress']
        else:
            self.playerfield[idx_x, idx_y] = self.minefield[idx_x, idx_y]
            num_hidden_tiles = np.count_nonzero(self.playerfield == -1)
            guess = True
            if self.move_num != 0:
                for k in range(-1, 2):
                    for h in range(-1, 2):
                        if k != 0 or h != 0:
                            idx1 = idx_x + k
                            idx2 = idx_y + h
                            if 0 <= idx1 < self.nrows and 0 <= idx2 < self.ncols:
                                if self.playerfield[idx1, idx2] != -1:
                                    guess = False
                                    break
            if self.playerfield[idx_x, idx_y] == -2:
                # Tile was a hidden mine, game over
                self.done = True
                self.explosion = True
                reward = self.reward['lose']
            elif num_hidden_tiles == self.mine_count:
                # The player has won by revealing all non-mine tiles
                self.done = True
                reward = self.reward['win']
                self.score += 1
            elif self.playerfield[idx_x, idx_y] == 0:
                self.score += 1
                # The tile was a zero, run auto-reveal routine
                self.auto_reveal_tiles(action)
                num_hidden_tiles = np.count_nonzero(self.playerfield == -1)
                if num_hidden_tiles == self.mine_count:
                    self.done = True
                    reward = self.reward['win']
                else:
                    if guess:
                        reward = self.reward['guess']
                    else:
                        reward = self.reward['progress']
            else:
                # Player has revealed a non-mine tile, but has not won yet
                if guess:
                    reward = self.reward['guess']
                else:
                    reward = self.reward['progress']
                self.score += 1
        # Update environment parameters
        self.move_num += 1
        return self.get_state(), reward, self.done

    def reset(self):
        # Resets all class variables to initial values, generates a new 
        # Minesweeper game, plays the first move, and returns the state
        self.score = 0
        self.move_num = 0
        self.explosion = False
        self.done = False
        self.minefield = np.zeros((self.nrows, self.ncols), dtype='int')
        self.playerfield = np.ones((self.nrows, self.ncols), dtype='int') * -1
        return self.play_first_move()

    def get_state(self):
        state = np.expand_dims(self.playerfield, axis=0)
        state = state / 8
        return state

    def generate_field(self, action):
        # Generates the minefield using the seeded random number generator.
        # The while loop randomly places mines in the grid, and then increments
        # the tile number of all adjacent non-mine tiles
        idx_x, idx_y = np.unravel_index(action, (self.nrows, self.ncols))
        num_mines = 0
        while num_mines < self.mine_count:
            x_rand = self.np_random.randint(0, self.nrows)
            y_rand = self.np_random.randint(0, self.ncols)
            # Reserve a mine-free tile for the player's first move
            if (x_rand, y_rand) != (idx_x, idx_y):
                if self.minefield[x_rand, y_rand] != -2:
                    self.minefield[x_rand, y_rand] = -2
                    num_mines += 1
                    for k in range(-1, 2):
                        for h in range(-1, 2):
                            if 0 <= x_rand + k < self.nrows and 0 <= y_rand + h < self.ncols:
                                if self.minefield[x_rand+k, y_rand+h] != -2:
                                    self.minefield[x_rand+k, y_rand+h] += 1

    def play_first_move(self):
        # The first move is automatically played by the environment, and is 
        # guaranteed to not contain a mine. Assign the value of this tile to 
        # the game state
        action_idx = self.np_random.randint(0, self.nrows * self.ncols)
        state, reward, done = self.step(action_idx)
        return state

    def seed(self, seed=None):
        self.np_random.seed(seed)

    def auto_reveal_tiles(self, action):
        # If the player selects a safe tile that has no adjacent mines (a zero)
        # all adjacent tiles will be revealed, and any zero tiles revealed 
        # will also have their adjacent tiles revealed in a chain reaction
        idx_x, idx_y = np.unravel_index(action, (self.nrows, self.ncols))
        for k in range(-1, 2):
            for h in range(-1, 2):
                idx1 = idx_x + k
                idx2 = idx_y + h
                if 0 <= idx1 < self.nrows and 0 <= idx2 < self.ncols:
                    if self.playerfield[idx1, idx2] == -1:
                        val = self.minefield[idx1, idx2]
                        self.playerfield[idx1, idx2] = val
                        self.score += 1
                        if val == 0:
                            self.auto_reveal_tiles(idx1*self.ncols + idx2)

    def init_gui(self):
        # Initialize all PyGame and GUI parameters
        pygame.init()
        pygame.mixer.quit()  # Fixes bug with high PyGame CPU usage
        self.tile_rowdim = 32  # pixels per tile along the horizontal
        self.tile_coldim = 32  # pixels per tile along the vertical
        self.game_width = self.ncols * self.tile_coldim
        self.game_height = self.nrows * self.tile_rowdim
        # self.ui_height = 32  # Contains text regarding score and move #
        self.gameDisplay = pygame.display.set_mode((self.game_width, self.game_height))
        pygame.display.set_caption('Minesweeper')
        # Load Minesweeper tileset
        self.tilemine = spr_mine.convert()
        self.tile0 = spr_emptyGrid.convert()
        self.tile1 = spr_grid1.convert()
        self.tile2 = spr_grid2.convert()
        self.tile3 = spr_grid3.convert()
        self.tile4 = spr_grid4.convert()
        self.tile5 = spr_grid5.convert()
        self.tile6 = spr_grid6.convert()
        self.tile7 = spr_grid7.convert()
        self.tile8 = spr_grid8.convert()
        self.tilehidden = spr_grid.convert()
        self.tileexplode = spr_mineFalse.convert()
        self.tile_dict = {-1: self.tilehidden, 0: self.tile0, 1: self.tile1,
                          2: self.tile2, 3: self.tile3, 4: self.tile4, 5: self.tile5,
                          6: self.tile6, 7: self.tile7, 8: self.tile8,
                          -2: self.tileexplode}
        # Set font and font color
        self.myfont = pygame.font.SysFont('Segoe UI', 32)
        self.font_color = (255, 255, 255)  # White
        self.victory_color = (8, 212, 29)  # Green
        self.defeat_color = (255, 0, 0)  # Red
        # Create selection surface to show what tile the agent is choosing
        self.selectionSurface = pygame.Surface((self.tile_rowdim, self.tile_coldim))
        self.selectionSurface.set_alpha(128)  # Opacity from 255 (opaque) to 0 (transparent)
        self.selectionSurface.fill((245, 245, 66))  # Yellow

    def render(self):
        # Update the game display after every agent action
        # Accepts a masked array of Q-values to plot as an overlay on the GUI
        # Update and blit text
        self.gameDisplay.fill(pygame.Color('black'))  # Clear screen
        self.plot_playerfield()
        self.update_screen()

    def plot_playerfield(self):
        # Blits the current state's tiles onto the game display
        for k in range(0, self.nrows):
            for h in range(0, self.ncols):
                self.gameDisplay.blit(self.tile_dict[self.playerfield[k, h]], (h*self.tile_coldim, k*self.tile_rowdim))
    
    def selection_animation(self, action):
        # Blits a transparent yellow rectangle over the tile the agent intends
        # to select
        row_idx, col_idx = np.unravel_index(action, (self.nrows, self.ncols))
        self.gameDisplay.blit(self.selectionSurface, (col_idx*self.tile_coldim, row_idx*self.tile_rowdim))

    def plot_minefield(self, action=None):
        # Plots the true minefield state that is hidden from the player
        # If an action is supplied only blits the mines for the final game view
        if action:
            # Plot location of mines at end of game
            row_idx, col_idx = np.unravel_index(action, (self.nrows, self.ncols))
            for k in range(0, self.nrows):
                for h in range(0, self.ncols):
                    if self.minefield[k, h] == -1:
                        # Only blit mines
                        self.gameDisplay.blit(self.tile_dict[self.minefield[k, h]],
                                              (h*self.tile_coldim, k*self.tile_rowdim))
            # Plot game-ending mine with red background color
            if self.explosion:
                self.gameDisplay.blit(self.tileexplode, (col_idx*self.tile_coldim, row_idx*self.tile_rowdim))
        else:
            # Plot for debug purposes        
            for k in range(0, self.nrows):
                for h in range(0, self.ncols):
                    self.gameDisplay.blit(self.tile_dict[self.minefield[k, h]],
                                          (h*self.tile_coldim, k*self.tile_rowdim))
        self.update_screen()

    def update_screen(self):
        pygame.display.update()
    
    def close(self):
        pygame.quit()
