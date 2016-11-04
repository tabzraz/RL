import gym
from gym import spaces
import pygame
import numpy as np


class MazeEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"]}
    # (0,0) is top left
    ACTIONS = ["W", "S", "E", "N"]

    def reset_maze(self):
        size = self.maze.shape
        x_size = size[0] - 1
        y_size = size[1] - 1
        # Walls at the edge
        self.maze[0, :] = 1
        self.maze[x_size, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, y_size] = 1

        # Player starts at bottom left corner
        self.maze[1, 1] = 3
        self.player_pos = (1, 1)

        # Goal in top right corner
        self.maze[x_size - 1, y_size - 1] = 2

        self.made_screen = False

    def __init__(self, size=(10, 10)):
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        self.maze = np.zeros(shape=size)

        self.reset_maze()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=3, shape=size)
        self.reward_range = (-1, 10)

    def _step(self, a):
        new_player_pos = (self.player_pos[0] + self.actions[a][0], self.player_pos[1] + self.actions[a][1])
        r = 0
        finished = False
        # Into a wall => negative reward
        if self.maze[new_player_pos] == 1:
            r = -1
            new_player_pos = self.player_pos
        elif self.maze[new_player_pos] == 2:
            r = +10
            finished = True
        self.maze[self.player_pos] = 0
        self.maze[new_player_pos] = 3
        self.player_pos = new_player_pos

        return self.maze, r, finished, {}

    def _reset(self):
        self.reset_maze()
        return self.maze

    def _render(self, mode="human", close=False):
        if close:
            pygame.quit()
            return
        if not self.made_screen:
            pygame.init()
            screen_size = (self.maze.shape[0] * 20, self.maze.shape[1] * 20)
            screen = pygame.display.set_mode(screen_size)
            self.screen = screen
            self.made_screen = True

        self.screen.fill((0, 0, 0))
        maze = self.maze

        for x in range(maze.shape[0]):
            for y in range(maze.shape[1]):
                if maze[x, y] != 0:
                    if maze[x, y] == 1:
                        colour = (255, 0, 0)
                    elif maze[x, y] == 2:
                        colour = (0, 255, 0)
                    elif maze[x, y] == 3:
                        colour = (0, 0, 255)
                    pygame.draw.rect(self.screen, colour, (y * 20, x * 20, 20, 20))

        pygame.display.update()
