from .GridWorld import GridWorld
import numpy as np


class BigRoom(GridWorld):

    def __init__(self, size):
        self.size = size
        super().__init__()
        print("Room of size:", self.grid.shape)

    def create_grid(self):
        self.grid = np.ones(shape=(self.size, self.size))
        self.grid[1:-1, 1:-1] = 0
        # self.grid[self.size // 2, 1] = 1
        # Goal
        self.grid[1, self.size // 2] = 2
        self.grid[-2, self.size // 2] = 2
        # self.grid[0,0] = 2
        self.grid[self.size // 2, 1] = 2
        # self.grid[self.size // 2, -2] = 2
        # Player
        self.grid[self.size // 2, self.size // 2] = 3
