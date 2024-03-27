import pygame
import random
pygame.init()


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


# Create class grid
class Grid:
    def __init__(self, xGrid, yGrid, val, game):
        self.xGrid = xGrid  # X pos of grid
        self.yGrid = yGrid  # Y pos of grid
        self.clicked = False  # Boolean var to check if the grid has been clicked
        self.mineClicked = False  # Bool var to check if the grid is clicked and it's a mine
        self.mineFalse = False  # Bool var to check if the player flagged the wrong grid
        self.flag = False  # Bool var to check if player flagged the grid
        # Create rectObject to handle drawing and collisions
        self.rect = pygame.Rect(game.border + self.xGrid * game.grid_size,
                                game.top_border + self.yGrid * game.grid_size, game.grid_size, game.grid_size)
        self.val = val  # Value of the grid, -1 is mine
        self.free = 0  # Number of free grid around the grid
        self.game = game

    def drawGrid(self):
        # Draw the grid according to bool variables and value of grid
        if self.mineFalse:
            self.game.gameDisplay.blit(spr_mineFalse, self.rect)
        else:
            if self.clicked:
                if self.val == -1:
                    if self.mineClicked:
                        self.game.gameDisplay.blit(spr_mineClicked, self.rect)
                    else:
                        self.game.gameDisplay.blit(spr_mine, self.rect)
                else:
                    if self.val == 0:
                        self.game.gameDisplay.blit(spr_emptyGrid, self.rect)
                    elif self.val == 1:
                        self.game.gameDisplay.blit(spr_grid1, self.rect)
                    elif self.val == 2:
                        self.game.gameDisplay.blit(spr_grid2, self.rect)
                    elif self.val == 3:
                        self.game.gameDisplay.blit(spr_grid3, self.rect)
                    elif self.val == 4:
                        self.game.gameDisplay.blit(spr_grid4, self.rect)
                    elif self.val == 5:
                        self.game.gameDisplay.blit(spr_grid5, self.rect)
                    elif self.val == 6:
                        self.game.gameDisplay.blit(spr_grid6, self.rect)
                    elif self.val == 7:
                        self.game.gameDisplay.blit(spr_grid7, self.rect)
                    elif self.val == 8:
                        self.game.gameDisplay.blit(spr_grid8, self.rect)

            else:
                if self.flag:
                    self.game.gameDisplay.blit(spr_flag, self.rect)
                else:
                    self.game.gameDisplay.blit(spr_grid, self.rect)

    def revealGrid(self):
        self.clicked = True
        # Auto reveal if it's a 0
        if self.val == 0:
            for x in range(-1, 2):
                if self.xGrid + x >= 0 and self.xGrid + x < self.game.game_width_height:
                    for y in range(-1, 2):
                        if self.yGrid + y >= 0 and self.yGrid + y < self.game.game_width_height:
                            if not self.game.grid[self.yGrid + y][self.xGrid + x].clicked:
                                self.game.grid[self.yGrid + y][self.xGrid + x].revealGrid()
            return 10
        elif self.val == -1:
            # Auto reveal all mines if it's a mine
            for m in self.game.mines:
                if not self.game.grid[m[1]][m[0]].clicked:
                    self.game.grid[m[1]][m[0]].revealGrid()
            return - 10
        return 0

    def updateValue(self):
        # Update the value when all grid is generated
        if self.val != -1:
            for x in range(-1, 2):
                if self.xGrid + x >= 0 and self.xGrid + x < self.game.game_width_height:
                    for y in range(-1, 2):
                        if self.yGrid + y >= 0 and self.yGrid + y < self.game.game_width_height:
                            if self.game.grid[self.yGrid + y][self.xGrid + x].val == -1:
                                self.val += 1
                            if self.game.grid[self.yGrid + y][self.xGrid + x].clicked:
                                self.free += 1


BG_COLOR = (192, 192, 192)
GRID_COLOR = (128, 128, 128)
GRID_SIZE = 32
BORDER = 16
TOP_BORDER = 100
CAPTION = "Minesweeper"


class Game:
    def __init__(self, width_height=None, num_mine=None):
        self.bg_color = BG_COLOR
        self.grid_color = GRID_COLOR

        self.game_width_height = width_height if width_height else 20  # Change this to increase size
        self.numMine = num_mine if num_mine else 40  # Number of mines
        self.grid_size = GRID_SIZE  # Size of grid (WARNING: macke sure to change the images dimension as well)
        self.border = BORDER  # Top border
        self.top_border = TOP_BORDER  # Left, Right, Bottom border
        self.display_width = self.grid_size * self.game_width_height + self.border * 2  # Display width
        self.display_height = self.grid_size * self.game_width_height + self.border + self.top_border  # Display height
        self.gameDisplay = pygame.display.set_mode((self.display_width, self.display_height))  # Create display
        self.timer = pygame.time.Clock()  # Create timer
        self.game_state = "Playing"
        self.mine_left = self.numMine
        self.grid = []
        self.mines = []
        self.t = 0  # Set time to 0
        pygame.display.set_caption(CAPTION)  # S Set the caption of window

    def reset(self):
        self.mine_left = self.numMine
        self.game_state = "Playing"
        self.grid = []
        self.mines = []
        self.t = 0
        self.timer = pygame.time.Clock()

        # Generating mines
        self.mines = [[random.randrange(0, self.game_width_height),
                       random.randrange(0, self.game_width_height)]]

        for c in range(self.numMine - 1):
            pos = [random.randrange(0, self.game_width_height),
                   random.randrange(0, self.game_width_height)]
            same = True
            while same:
                for i in range(len(self.mines)):
                    if pos == self.mines[i]:
                        pos = [random.randrange(0, self.game_width_height),
                               random.randrange(0, self.game_width_height)]
                        break
                    if i == len(self.mines) - 1:
                        same = False
            self.mines.append(pos)

        # Generating entire grid
        for j in range(self.game_width_height):
            line = []
            for i in range(self.game_width_height):
                if [i, j] in self.mines:
                    line.append(Grid(i, j, -1, self))
                else:
                    line.append(Grid(i, j, 0, self))
            self.grid.append(line)

        # Update of the grid
        for i in self.grid:
            for j in i:
                j.updateValue()
        # self._update_ui()

    def play_step(self, action):
        self.gameDisplay.fill(self.bg_color)  # Fill the screen with bg color

        reward = 0
        x = action[0]
        y = action[1]
        left_click = action[2] == 0
        right_click = action[2] == 1
        selected = self.grid[y][x]
        if left_click:
            # If player left-clicked of the grid
            reward += selected.revealGrid()
            # Toggle flag off
            if selected.flag:
                self.mine_left += 1
                selected.flag = False
            # If it's a mine
            if selected.val == -1:
                self.game_state = "Game Over"
                selected.mineClicked = True
        elif right_click:
            # If the player right-clicked
            if not selected.clicked:
                if selected.flag:
                    selected.flag = False
                    self.mine_left += 1
                else:
                    selected.flag = True
                    self.mine_left -= 1
        # Check if won
        w = True
        for i in self.grid:
            for j in i:
                j.drawGrid()
                if j.val != -1 and not j.clicked:
                    w = False
        if w and self.game_state != "Exit":
            self.game_state = "Win"

        # Draw Texts
        if self.game_state != "Game Over" and self.game_state != "Win":
            self.t += 1
        self._update_ui()
        done = self.game_state != "Playing"
        return reward, done, self.numMine - self.mine_left

    # # Create function to draw texts
    # def _drawText(self, txt, s, yOff=0):
    #     screen_text = pygame.font.SysFont("Calibri", s, True).render(txt, True, (0, 0, 0))
    #     rect = screen_text.get_rect()
    #     rect.center = (game_width * grid_size / 2 + border, game_height * grid_size / 2 + top_border + yOff)
    #     gameDisplay.blit(screen_text, rect)

    def _update_ui(self):
        s = str(self.t // 15)
        screen_text = pygame.font.SysFont("Calibri", 50).render(s, True, (0, 0, 0))
        self.gameDisplay.blit(screen_text, (self.border, self.border))
        # Draw mine left
        screen_text = pygame.font.SysFont("Calibri", 50).render(self.mine_left.__str__(),
                                                                True, (0, 0, 0))
        self.gameDisplay.blit(screen_text, (self.display_width - self.border - 50, self.border))

        pygame.display.update()  # Update screen

        # self.timer.tick(15)  # Tick fps


if __name__ == "__main__":
    game_1 = Game()
    game_1.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        game_1.play_step((9, 1, 1))
        # pygame.display.update()
        game_1.timer.tick(15)
