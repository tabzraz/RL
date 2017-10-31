import numpy as np
import gym
from gym import spaces


class Mario(gym.Env):

    # Our wrapper handles the drawing
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self):
        self.mario_env = gym.make("ppaquette/SuperMarioBros-1-1-v0")
        self.mario_x = 0

    def _step(self, a):
        s, r, finished, info_dict = self.mario_env.step(a)
        if info_dict["distance"] >= 3200:
            # We have reached the flagpole
            r = +1
        else:
            r = -0.0001
        self.mario_x = info_dict["distance"]
        return s, r, finished, info_dict

    def _reset(self):
        return self.mario_env.reset()

    def _render(self, mode="rgb_array", close=False):
        return self.mario_env.render(mode=mode, close=close)

    def log_player_pos(self):
        return int(self.mario_x)

    def trained_on_states(self, player_visits, args):
        pass

    def xp_and_frontier_states(self):
        pass

    def bonus_xp_and_frontier_states(self):
        pass

    def visits_and_frontier_states(self):
        pass

    def xp_replay_states(self, player_visits, args, bonus_replay=False):
        pass

    def player_visits(self, player_visits, args):
        pass

    def bonus_landscape(self, player_visits, exploration_bonuses, max_bonus, args):
        pass

    def frontier(self, exp_model, args, max_bonus=None):
        pass
