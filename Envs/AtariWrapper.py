import numpy as np
import gym
from .OpenAI_AtariWrapper import wrap_deepmind


class AtariEnv(gym.Env):
    # Gotta include this to render
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name="Breakout", colours=True, history_length=4, resized_size=(42, 42), action_repeat=4):
        env_name = "{}NoFrameskip-v4".format(game_name)
        self.env = gym.make(env_name)
        self.env = wrap_deepmind(self.env)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.ale = self.env.unwrapped.ale
        ale_ram_size = self.ale.getRAMSize()
        self.ram = np.zeros((ale_ram_size), dtype=np.uint8)
        # (Screen, x, y)
        self.position = (0, 0, 0)

    def _step(self, a):
        s, r, finished, info_dict = self.env.step(a)
        # Log the position for Montezuma. TODO: Move this elsewhere
        self.ale.getRAM(self.ram)
        self.position = (self.ram[3], self.ram[42], self.ram[43])
        return s, r, finished, info_dict

    def _reset(self):
        return self.env.reset()

    def _render(self, mode="rgb_array", close=False):
        return self.env.render(mode=mode, close=close)

    def log_player_pos(self):
        return self.position

    # At the moment this is just to save the player positions
    def player_visits(self, player_visits, args):
        # Log the visitations
        with open("{}/logs/Player_Positions.txt".format(args.log_path), "a") as file:
            file.write('\n'.join(" ".join(str(x) for x in t) for t in player_visits))

        return None

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

    def bonus_landscape(self, player_visits, exploration_bonuses, max_bonus, args):
        pass

    def frontier(self, exp_model, args, max_bonus=None):
        pass
