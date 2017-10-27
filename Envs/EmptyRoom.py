from .GridWorld import GridWorld
import numpy as np


class EmptyRoom(GridWorld):

    def __init__(self, size):
        self.size = size
        # self.rnd_seed = rnd_seed
        # self.corridor_width = corridor_width
        super().__init__()
        print("Empty Room of size:", self.grid.shape)

        # Count number of states
        num_states = 0
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                if self.grid[x, y] != 1:
                    num_states += 1

        print("Number of states: {}".format(num_states))

        self.negative_reward = 0

    def create_grid(self):
        self.grid = np.ones(shape=(self.size * 3, self.size * 3))
        self.grid[1:-1, 1:-1] = 0
        # Player
        center_of_room = (self.size * 3) // 2
        self.grid[center_of_room, center_of_room] = 3
        # Goal
        # We need 1 goal to not break the gridworld environment
        self.grid[0, 0] = 2
