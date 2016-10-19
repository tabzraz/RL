import numpy as np
import gym
from skimage.transform import resize
from skimage.color import rgb2grey  # Grey because we speak the Queen's English


class AtariEnv(gym.Env):
    # Gotta include this to render
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name="Breakout", colours=True, history_length=4, resized_size=(105, 80), action_repeat=4):
        env_name = "{}NoFrameskip-v3".format(game_name)
        self.env = gym.make(env_name)
        self.colours = colours
        self.resized_size = resized_size
        self.history_length = history_length
        self.action_repeat = action_repeat

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # Gym Atari env dosen't seem to set reward_range

        self.frames = self.start_frames(self.env.reset())

    def add_frame(self, new_frame):
        if not self.colours:
            new_frame = rgb2grey(new_frame)
        resized_frame = resize(new_frame, self.resized_size)
        self.frames = np.concatenate([self.frames[:, :, (self.history_length - 1):], resized_frame], axis=2)

    def start_frames(self, frame):
        if not self.colours:
            frame = rgb2grey(frame)
        resized_frame = resize(frame, self.resized_size)
        new_frames = np.concatenate([resized_frame for _ in range(self.history_length)], axis=2)
        return new_frames

    # Gym Env required methods:

    def _step(self, a):
        episode_finished = False
        r = 0
        for _ in range(self.action_repeat):
            if not episode_finished:
                s_t, r_t, episode_finished, info = self.env.step(a)
                r += r_t
            else:
                break
        self.add_frame(s_t)
        return self.frames, r, episode_finished, info

    def _reset(self):
        self.frames = self.start_frames(self.env.reset())
        return self.frames

    def _render(self, mode='human', close=False):
        self.env.render(mode=mode, close=close)

    def _seed(self, seed=None):
        return self.env.seed(seed)

    # Gym Atari dosen't implement the following:
    # _close
    # _configure
