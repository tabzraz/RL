from .GridWorld import GridWorld
import numpy as np


class BigRoom(GridWorld):

    def __init__(self, size):
        self.size = size
        super().__init__()
        print("Room of size:", self.grid.shape)

        # No negative reward at each timestep
        self.negative_reward = 0

    def create_grid(self):
        self.grid = np.ones(shape=(self.size, self.size))
        self.grid[1:-1, 1:-1] = 0
        # Goal
        self.grid[1, -2] = 2
        # Player
        self.grid[-2, 1] = 3
