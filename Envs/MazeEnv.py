from __future__ import division
import gym
from gym import spaces
import pygame
import numpy as np


class MazeEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"]}
    # (0,0) is top left
    ACTIONS = ["W", "S", "E", "N"]

    def reset_maze(self):
        self.steps = 0

        # 0 = Nothing
        # 1 = Wall
        # 2 = Goal
        # 3 = Player
        self.tile = \
        np.array([[1, 1, 1, 0, 1, 1, 1,],
                  [1, 0, 0, 0, 0, 0, 1,],
                  [1, 0, 1, 1, 0, 1, 1,],
                  [0, 0, 0, 1, 0, 0, 0,],
                  [1, 1, 0, 1, 1, 0, 1,],
                  [1, 0, 0, 0, 0, 0, 1,],
                  [1, 1, 1, 0, 1, 1, 1,],
                  ])

        self.maze = np.tile(self.tile, self.shape)
        # Player
        self.maze[0,3] = 3
        # Goal
        self.maze[-1, -4] = 2

        self.player_pos = (0, 3)
        self.made_screen = False
        self.pellets = (self.maze == 2).sum()

    def __init__(self, size=(1, 1), seed=2, normalise=True):
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        self.shape = size
        self.maze = np.zeros(shape=(size[0]*7, size[1]*7))
        self.seed = seed
        self.steps = 0
        self.limit = self.maze.size * 3
        self.negative_reward = -1 / self.limit
        self.positive_reward = +1

        self.reset_maze()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=3, shape=self.maze.shape)
        self.reward_range = (-1, 1)

    def _step(self, a):
        self.steps += 1
        new_player_pos = (self.player_pos[0] + self.actions[a][0], self.player_pos[1] + self.actions[a][1])
        # Clip
        if new_player_pos[0] < 0 or new_player_pos[0] >= self.maze.shape[0]\
        or new_player_pos[1] < 0 or new_player_pos[1] >= self.maze.shape[1]:
            new_player_pos = self.player_pos

        r = self.negative_reward  # -1 on every step normalizedish
        finished = False
        # Into a wall => negative reward
        if self.maze[new_player_pos] == 1:
            # r += -5
            new_player_pos = self.player_pos
        elif self.maze[new_player_pos] == 2:
            r += self.positive_reward
            self.pellets -= 1
            if self.pellets == 0:
                finished = True
        self.maze[self.player_pos] = 0
        self.maze[new_player_pos] = 3
        self.player_pos = new_player_pos

        if self.steps >= self.limit:
            finished = True

        return self.maze[:, :, np.newaxis] / 3, r, finished, {}

    def _reset(self):
        self.reset_maze()
        return self.maze[:, :, np.newaxis] / 3

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
