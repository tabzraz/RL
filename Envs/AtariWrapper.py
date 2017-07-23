import numpy as np
import gym
# from skimage.transform import resize
from scipy.misc import imresize as resize
from skimage.color import rgb2grey  # Grey because we speak the Queen's English


class AtariEnv(gym.Env):
    # Gotta include this to render
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name="Breakout", colours=True, history_length=4, resized_size=(42, 42), action_repeat=4):
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
        resized_frame = resize(new_frame, self.resized_size, mode="F", interp="bilinear")
        # resize(state[:, :, 0], self.cts_model_shape, mode="F", interp="bilinear")
        if self.colours:
            self.frames = np.concatenate([self.frames[:, :, 3:], resized_frame], axis=2)
        else:
            self.frames = np.concatenate([self.frames[:, :, 1:], resized_frame[:, :, np.newaxis]], axis=2)

    def start_frames(self, frame):
        if not self.colours:
            frame = rgb2grey(frame)
        # resized_frame = resize(frame, self.resized_size)
        resized_frame = resize(frame, self.resized_size, mode="F", interp="bilinear")
        if self.colours:
            new_frames = np.concatenate([resized_frame for _ in range(self.history_length)], axis=2)
        else:
            new_frames = np.stack([resized_frame for _ in range(self.history_length)], axis=-1)
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
        # return self.env.render(mode=mode, close=close)
        if mode == "human":
            self.env.render(mode="human", close=close)
        # print(self.frames.shape)
        grid = self.frames[:, :, -1]
        # print(grid.shape)
        image = np.zeros(shape=(grid.shape[0], grid.shape[1], 3))
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y] != 0:
                    image[x, y] = (255 * grid[x, y], 255 * grid[x, y], 255 * grid[x, y])
        return image

    def _seed(self, seed=None):
        return self.env.seed(seed)

    # Gym Atari dosen't implement the following:
    # _close
    # _configure
