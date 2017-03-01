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
        # Goals
        self.maze[-1, -4] = 2  #Bottom Right
        self.maze[3, -1] = 2  #Top Right 
        self.maze[-4, 0] = 2  #Bottom Left
        # Player
        self.maze[0, 3] = 3

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
        info_dict = {}
        new_player_pos = (self.player_pos[0] + self.actions[a][0], self.player_pos[1] + self.actions[a][1])
        # Clip
        r = self.negative_reward
        if new_player_pos[0] < 0 or new_player_pos[0] >= self.maze.shape[0]\
           or new_player_pos[1] < 0 or new_player_pos[1] >= self.maze.shape[1]:
            new_player_pos = self.player_pos
            # r += self.negative_reward

        # r = 0  # -1 on every step normalizedish
        finished = False
        # Into a wall => negative reward
        if self.maze[new_player_pos] == 1:
            # r += -5
            # r += self.negative_reward
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
            info_dict["Steps_Termination"] = True

        return self.maze[:, :, np.newaxis] / 3, r, finished, info_dict

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
                    colour = (255 * maze[x, y] / 3, 255 * maze[x, y] / 3, 255 * maze[x, y] / 3)
                    # colour = (255 - colour[0], 255 - colour[1], 255 - colour[2])
                    pygame.draw.rect(self.screen, colour, (y * 20, x * 20, 20, 20))

        # for x in range(maze.shape[0]):
        #     for y in range(maze.shape[1]):
        #         if maze[x, y] != 0:
        #             if maze[x, y] == 1:
        #                 colour = (255, 0, 0)
        #             elif maze[x, y] == 2:
        #                 colour = (0, 255, 0)
        #             elif maze[x, y] == 3:
        #                 colour = (0, 0, 255)
        #             pygame.draw.rect(self.screen, colour, (y * 20, x * 20, 20, 20))

        pygame.display.update()

    def debug_render(self, debug_info=None, close=False):
        if close:
            pygame.quit()
            return
        if not self.made_screen:
            pygame.init()
            screen_size = (self.maze.shape[0] * 20 + 60, self.maze.shape[1] * 20 + 120)
            screen = pygame.display.set_mode(screen_size)
            self.screen = screen
            self.made_screen = True

        self.screen.fill((0, 0, 0))
        maze = self.maze

        for x in range(maze.shape[0]):
            for y in range(maze.shape[1]):
                if maze[x, y] != 0:
                    colour = (255 * maze[x, y] / 3, 255 * maze[x, y] / 3, 255 * maze[x, y] / 3)
                    # colour = (255 - colour[0], 255 - colour[1], 255 - colour[2])
                    pygame.draw.rect(self.screen, colour, (y * 20, x * 20, 20, 20))

        if debug_info is None:
            return

        white_colour = (255, 255, 255)
        red_colour = (255, 0, 0)
        blue_colour = (0, 0, 255)

        if "Exp_Bonus" in debug_info:
            exploration_bonus = debug_info["Exp_Bonus"]
            max_exp_bonus = debug_info["Max_Exp_Bonus"]
            # Exploration_Bonus
            exp_bonus_size = int(exploration_bonus / max_exp_bonus * 40)
            pygame.draw.rect(self.screen, red_colour, (self.maze.shape[0] * 20 + 10, 10 + (self.maze.shape[1] * 20) + 100 - exp_bonus_size, 40, exp_bonus_size), 0)
            pygame.draw.rect(self.screen, white_colour, (self.maze.shape[0] * 20 + 10, 10, 40, (self.maze.shape[1] * 20) + 100), 2)

        if "Q_Values" in debug_info:
            q_values = debug_info["Q_Values"]
            max_q_value = debug_info["Max_Q_Value"]

            # Q vals
            q_val_sizes = [int(q_val / max_q_value * 100) for q_val in q_values]
            actions = len(q_values)
            for i, q_size in enumerate(q_val_sizes):
                pygame.draw.rect(self.screen, blue_colour, (10 + int((self.maze.shape[0] * 20 - 20) / actions) * i, self.maze.shape[1] * 20 + 10 + 100 - q_size, int((self.maze.shape[0] * 20 - 20) / 4), q_size), 0)
                pygame.draw.rect(self.screen, white_colour, (10 + int((self.maze.shape[0] * 20 - 20) / actions) * i, self.maze.shape[1] * 20 + 10 + 100 - q_size, int((self.maze.shape[0] * 20 - 20) / 4), q_size), 2)

        pygame.display.update()
