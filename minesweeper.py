#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:02:40 2019

@author: alex

###############################################################################
# minesweeper.py
#
# Revision:     1.00
# Date:         6/03/2020
# Author:       Alex
#
# Purpose:      Contains all functions necessary to implement the Minesweeper 
#               environment, including a PyGame GUI.
#
##############################################################################
"""


import pygame
import numpy as np
from collections import deque


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

    def __init__(self, rowdim, coldim, mine_count, gui=False):
        self.rowdim = rowdim  # number of tiles along the row dimension
        self.coldim = coldim  # number of tiles along the column dimension
        self.mine_count = mine_count
        self.minefield = np.zeros((rowdim, coldim), dtype='int')  # The complete game state
        self.playerfield = np.ones((rowdim, coldim), dtype='int')*9  # The state the player sees
        self.explosion = False  # True if player selects mine
        self.done = False  # Game complete (win or loss)
        self.score = 0
        self.np_random = np.random.RandomState()  # For seeding the environment
        self.move_num = 0  # Track number of player moves per game
        # self.win = False
        if gui:
            self.init_gui()  # Pygame related parameters

    def step(self, action):
        # Function accepts the player's action as an input and returns the new
        # environment state, a reward, and whether the episode has terminated
        if self.move_num == 0:
            self.generate_field(action)
        state = self.playerfield.flatten()
        minefield_state = self.minefield.flatten()
        if state[action] == minefield_state[action]:
            # Tile has already been revealed
            reward = -0.1
            score = 0
            done = False
        else:
            state[action] = minefield_state[action]
            num_hidden_tiles = np.count_nonzero(state == 9)
            if state[action] == -1:
                # Tile was a hidden mine, game over
                done = True
                self.explosion = True
                reward = -1.0
                score = 0  # Hitting mine should not subtract points from score
            elif num_hidden_tiles == self.mine_count:
                # The player has won by revealing all non-mine tiles
                done = True
                reward = 1.0
                score = 1
            elif state[action] == 0:
                # The tile was a zero, run auto-reveal routine
                state, score = self.auto_reveal_tiles(action)
                num_hidden_tiles = np.count_nonzero(state == 9)
                if num_hidden_tiles == self.mine_count:
                    done = True
                    reward = 1.0
                else:
                    done = False
                    reward = 0.2
            else:
                # Player has revealed a non-mine tile, but has not won yet
                done = False
                reward = 0.2
                score = 1
        # Update environment parameters
        state = state.reshape(self.rowdim, self.coldim)
        self.playerfield = state
        self.score += score 
        self.done = done
        self.move_num += 1
        return np.expand_dims(state, 0), reward, done

    def reset(self):
        # Resets all class variables to initial values, generates a new 
        # Minesweeper game, plays the first move, and returns the state
        self.score = 0
        self.move_num = 0
        self.explosion = False
        self.done = False
        self.win = False
        self.minefield = np.zeros((self.rowdim, self.coldim), dtype='int')
        self.playerfield = np.ones((self.rowdim, self.coldim), dtype='int')*9
        state = self.play_first_move()
        return state
        
    def generate_field(self, action):
        # Generates the minefield using the seeded random number generator.
        # The while loop randomly places mines in the grid, and then increments
        # the tile number of all adjacent non-mine tiles
        idx_x, idx_y = np.unravel_index(action, (self.rowdim, self.coldim))
        num_mines = 0
        while num_mines < self.mine_count:
            x_rand = self.np_random.randint(0,self.rowdim)
            y_rand = self.np_random.randint(0,self.coldim)
            # Reserve a mine-free tile for the player's first move
            if (x_rand, y_rand) != (idx_x, idx_y):
                if self.minefield[x_rand, y_rand] != -1:
                    self.minefield[x_rand, y_rand] = -1
                    num_mines += 1
                    for k in range(-1, 2):
                        for h in range(-1, 2):
                            try:
                                if self.minefield[x_rand+k, y_rand+h] != -1:
                                    if x_rand+k > -1 and y_rand+h > -1:
                                        self.minefield[x_rand+k, y_rand+h] += 1
                            except IndexError:
                                pass

    def play_first_move(self):
        # The first move is automatically played by the environment, and is 
        # guaranteed to not contain a mine. Assign the value of this tile to 
        # the game state
        action_idx = self.np_random.randint(0, self.rowdim*self.coldim-1)
        state, reward, done = self.step(action_idx)
        return state

    def seed(self, seed=None):
        self.np_random.seed(seed)

    def auto_reveal_tiles(self, action):
        # If the player selects a safe tile that has no adjacent mines (a zero)
        # all adjacent tiles will be revealed, and any zero tiles revealed 
        # will also have their adjacent tiles revealed in a chain reaction
        idx1, idx2 = np.unravel_index(action, (self.rowdim, self.coldim))
        old_zeros = []  # Keep track of already revealed zeros
        new_zeros = deque([(idx1, idx2)])  # Keep track of newly discovered zeros
        state = self.playerfield.copy()
        revealed_tiles = [tuple(x) for x in np.argwhere(state != 9)]  # Keep track of indices of revealed tiles
        score = 0
        zero_found = True
        while zero_found:
            # Iterate through indices of tile and its 8 neighbors
            for k in range(-1, 2):
                for h in range(-1, 2):
                    idx1 = new_zeros[0][0] + k
                    idx2 = new_zeros[0][1] + h
                    if idx1 >= 0 and idx2 >= 0:  # Avoid negative indices
                        try:
                            if (idx1, idx2) not in revealed_tiles:
                                # Reveal tile
                                state[idx1, idx2] = self.minefield[idx1, idx2]
                                score += 1
                                # Add revealed tile index to list
                                revealed_tiles.append((idx1, idx2))
                                # Check to see if it's also a zero tile
                                if self.minefield[idx1, idx2] == 0:
                                    # Make sure we haven't seen this zero before
                                    if (idx1, idx2) not in old_zeros:
                                        if (idx1, idx2) not in new_zeros:
                                            # Add to newly discovered zero list
                                            new_zeros.append((idx1, idx2))
                        except IndexError:
                            pass  # Ignore invalid indices at border of minefield
            # Move indices from new zero list to old zero list
            old_zeros.append(new_zeros.popleft())
            if len(new_zeros) == 0:
                # Terminate loop
                zero_found = False
            
        return state.flatten(), score

    def init_gui(self):
        # Initialize all PyGame and GUI parameters
        pygame.init()
        pygame.mixer.quit()  # Fixes bug with high PyGame CPU usage
        self.tile_rowdim = 32  # pixels per tile along the horizontal
        self.tile_coldim = 32  # pixels per tile along the vertical
        self.game_width = self.coldim * self.tile_coldim
        self.game_height = self.rowdim * self.tile_rowdim
        self.ui_height = 32  # Contains text regarding score and move #
        self.gameDisplay = pygame.display.set_mode((self.game_width, self.game_height+self.ui_height))
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
        self.tile_dict = {-1: self.tilemine, 0: self.tile0, 1: self.tile1,
                          2: self.tile2, 3: self.tile3, 4: self.tile4, 5: self.tile5,
                          6: self.tile6, 7: self.tile7, 8: self.tile8,
                          9: self.tilehidden, -2: self.tileexplode}
        # Set font and font color
        self.myfont = pygame.font.SysFont('Segoe UI', 32)
        self.font_color = (255, 255, 255)  # White
        self.victory_color = (8, 212, 29)  # Green
        self.defeat_color = (255, 0, 0)  # Red
        # Create selection surface to show what tile the agent is choosing
        self.selectionSurface = pygame.Surface((self.tile_rowdim, self.tile_coldim))
        self.selectionSurface.set_alpha(128)  # Opacity from 255 (opaque) to 0 (transparent)
        self.selectionSurface.fill((245, 245, 66))  # Yellow

    def render(self, valid_qvalues=np.array([])):
        # Update the game display after every agent action
        # Accepts a masked array of Q-values to plot as an overlay on the GUI
        # Update and blit text
        text_score = self.myfont.render('SCORE: ', True, self.font_color)
        text_score_number = self.myfont.render(str(self.score), True, self.font_color)
        text_move = self.myfont.render('MOVE: ', True, self.font_color)
        text_move_number = self.myfont.render(str(self.move_num), True, self.font_color)
        text_victory = self.myfont.render('VICTORY!', True, self.victory_color)
        text_defeat = self.myfont.render('DEFEAT!', True, self.defeat_color)
        self.gameDisplay.fill(pygame.Color('black'))  # Clear screen
        self.gameDisplay.blit(text_move, (40, self.game_height+0))
        self.gameDisplay.blit(text_move_number, (160, self.game_height+0))
        # self.gameDisplay.blit(text_score, (300, self.game_height+0))
        # self.gameDisplay.blit(text_score_number, (450, self.game_height+0))
        if self.done:
            if self.explosion:
                self.gameDisplay.blit(text_defeat, (700, self.game_height+5))
            else:
                self.gameDisplay.blit(text_victory, (700, self.game_height+5))
        # Blit updated view of minefield
        self.plot_playerfield()
        if valid_qvalues.size > 0:
            # Blit surface showing agent selection and Q-value representations
            self.selection_animation(np.argmax(valid_qvalues))
            self.plot_qvals(valid_qvalues)
        self.update_screen()

    def plot_playerfield(self):
        # Blits the current state's tiles onto the game display
        for k in range(0, self.rowdim):
            for h in range(0, self.coldim):
                self.gameDisplay.blit(self.tile_dict[self.playerfield[k, h]], (h*self.tile_coldim, k*self.tile_rowdim))
    
    def selection_animation(self, action):
        # Blits a transparent yellow rectangle over the tile the agent intends
        # to select
        row_idx, col_idx = np.unravel_index(action, (self.rowdim, self.coldim))
        self.gameDisplay.blit(self.selectionSurface, (col_idx*self.tile_coldim, row_idx*self.tile_rowdim))

    def plot_qvals(self, valid_qvalues):
        # Superimposes a colored circle over each unrevealed tile in the grid
        # A large blue circle is a tile the agent feels confident is safe
        # A large red circle is a tile the agent feels confident is a mine
        # A small dark/black colored circle is a tile the agent is unsure of
        max_qval = np.max(valid_qvalues)
        min_qval = np.min(valid_qvalues)
        qval_array = valid_qvalues.reshape(self.rowdim, self.coldim)
        for k in range(0, self.rowdim):
            for h in range(0, self.coldim):
                qval = qval_array[k, h]
                if not np.ma.is_masked(qval):  # Ignore revealed tiles
                    if qval >= 0:  # Color blue
                        qval_scale = np.abs((qval / max_qval) ** 0.5)
                        rgb_tuple = (0, 0, int(qval_scale*255))
                    else: # Color red
                        qval_scale = np.abs((qval / min_qval) ** 0.5)
                        rgb_tuple = (int(qval_scale*255), 0, 0)
                    center = (int(h*self.tile_coldim + self.tile_coldim/2),
                               int(k*self.tile_rowdim + self.tile_rowdim/2))
                    radius = int(self.tile_rowdim/4 * qval_scale)
                    pygame.draw.circle(self.gameDisplay, rgb_tuple, center, radius)

    def plot_minefield(self, action=None):
        # Plots the true minefield state that is hidden from the player
        # If an action is supplied only blits the mines for the final game view
        if action:
            # Plot location of mines at end of game
            row_idx, col_idx = np.unravel_index(action, (self.rowdim, self.coldim))
            for k in range(0, self.rowdim):
                for h in range(0, self.coldim):
                    if self.minefield[k, h] == -1:
                        # Only blit mines
                        self.gameDisplay.blit(self.tile_dict[self.minefield[k, h]], (h*self.tile_coldim, k*self.tile_rowdim))
            # Plot game-ending mine with red background color
            if self.explosion:
                self.gameDisplay.blit(self.tileexplode, (col_idx*self.tile_coldim, row_idx*self.tile_rowdim))
        else:
            # Plot for debug purposes        
            for k in range(0, self.rowdim):
                for h in range(0, self.coldim):
                    self.gameDisplay.blit(self.tile_dict[self.minefield[k, h]], (h*self.tile_coldim, k*self.tile_rowdim))
        self.update_screen()

    def update_screen(self):
        pygame.display.update()
    
    def close(self):
        pygame.quit()


'''
if __name__ == '__main__':
    # Test the Minesweeper environment
    env = Minesweeper(8, 8, 10, gui=True)
    env.reset()
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        action = np.random.randint(0, 64)
        state, reward, done = env.step(action)
        env.render()
        pygame.time.wait(500)
    env.plot_minefield(action)
    pygame.time.wait(5000)
    env.close()
'''
