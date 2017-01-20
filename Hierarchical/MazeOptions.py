import numpy as np


class MazeOptions:

    # 4 Actions available that take you to one of the 4 exits of the current room, and then try to exit that way.
    # Exit 0 is the left one, anti-clockwise from there
    # (0,0) is at the top left
    # (y, x) is the player_pos
    def __init__(self, maze_env):
        self.maze_env = maze_env
        self.action = 0
        self.steps = 0

        # Hard coding is best
        U = 2
        D = 3
        L = 4
        R = 5
        self.action_0 = \
        np.array([[1, 1, 1, D, 1, 1, 1,],
                  [1, D, L, L, L, L, 1,],
                  [1, D, 1, 1, U, 1, 1,],
                  [L, L, L, 1, U, L, L,],
                  [1, 1, U, 1, 1, D, 1,],
                  [1, R, U, L, L, L, 1,],
                  [1, 1, 1, U, 1, 1, 1,],
                  ])
        self.action_1 = \
        np.array([[1, 1, 1, D, 1, 1, 1,],
                  [1, D, L, R, D, L, 1,],
                  [1, D, 1, 1, D, 1, 1,],
                  [R, R, D, 1, R, D, L,],
                  [1, 1, D, 1, 1, D, 1,],
                  [1, R, R, D, L, L, 1,],
                  [1, 1, 1, D, 1, 1, 1,],
                  ])
        self.action_2 = \
        np.array([[1, 1, 1, D, 1, 1, 1,],
                  [1, R, R, R, R, L, 1,],
                  [1, U, 1, 1, D, 1, 1,],
                  [R, R, D, 1, R, R, R,],
                  [1, 1, D, 1, 1, U, 1,],
                  [1, R, R, R, R, U, 1,],
                  [1, 1, 1, U, 1, 1, 1,],
                  ])
        self.action_3 = \
        np.array([[1, 1, 1, U, 1, 1, 1,],
                  [1, R, R, U, L, L, 1,],
                  [1, U, 1, 1, U, 1, 1,],
                  [R, U, L, 1, U, L, L,],
                  [1, 1, U, 1, 1, U, 1,],
                  [1, R, U, L, R, U, 1,],
                  [1, 1, 1, U, 1, 1, 1,],
                  ])
        self.actions = [self.action_0, self.action_1, self.action_2, self.action_3]

    def choose_option(self, action):
        self.action = action
        self.steps = 0

    def act(self):
        self.steps += 1
        player_pos = self.maze_env
        player_pos[0] = player_pos[0] % 7
        player_pos[1] = player_pos[1] % 7
        action_to_take = self.actions[self.action][player_pos] - 2
        beta = 0.0
        if action_to_take == 0 and player_pos[1] == 0:
            beta = 1.0
        elif action_to_take == 1 and player_pos[0] == 6:
            beta = 1.0
        elif action_to_take == 2 and player_pos[1] == 6:
            beta = 1.0
        elif action_to_take == 3 and player_pos[1] == 0:
            beta = 1.0

        return action_to_take, beta
