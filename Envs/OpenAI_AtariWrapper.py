import numpy as np
from collections import deque
from PIL import Image
import gym
from gym import spaces


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4, max_over=2):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=max_over)
        self._skip       = skip

    def _step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ClipRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class ClipNegativeRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        if reward >= 0:
            return reward
        else:
            return 0

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, res=84):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.res = res
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.res, self.res, 1))

    def _observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
            resample=Image.BILINEAR), dtype=np.uint8)
        return frame.reshape((self.res, self.res, 1)) / 255.0

class ResizeFrame(gym.ObservationWrapper):
    def __init__(self, env, res=40):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.res = res
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.res, self.res, 1))

    def _observation(self, obs):
        frame = obs.astype("float32") * 255
        frame = frame[:, :, 0]
        # frame = np.concatenate([frame, frame, frame], axis=2)
        # frame = np.dot(frame, np.array([0.299, 0.587, 0.114], 'float32'))
        
        frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
            resample=Image.BILINEAR), dtype=np.uint8)
        return frame.reshape((self.res, self.res, 1)) / 255.0

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        assert shp[2] == 1  # can only stack 1-channel frames
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], k))

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k): self.frames.append(ob)
        return self._observation()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)


class GreyscaleRender(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def _render(self, mode="human", close=False):
        if mode == "human":
            self.unwrapped._render(mode=mode, close=close)
        else:
            # print(self.unwrapped)
            grid = self.env._observation()
            grid = grid[:, :, -1]
            grid = np.stack([grid for _ in range(3)], axis=2)
            return grid * 255

def wrap_maze(env):
    # Change the size of the maze to be (40, 40)
    env = ResizeFrame(env, res=40)
    env = FrameStack(env, 1)
    env = GreyscaleRender(env)
    print("Wrapping maze to be (40, 40)")
    return env

def wrap_deepmind(env, episode_life=True, clip_rewards=True, stack=4):
    """Configure environment for DeepMind-style Atari.

    Note: this does not include frame stacking!"""
    assert 'NoFrameskip' in env.spec.id  # required for DeepMind-style skip
    if episode_life:
        env = EpisodicLifeEnv(env)
    # env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if stack > 1:
        env = FrameStack(env, 4)
    print("Wrapping environment with Deepmind-style setttings.")
    env = GreyscaleRender(env)
    return env

resolutions = ['160x120', '200x125', '200x150', '256x144', '256x160', '256x192', '320x180', '320x200',
               '320x240', '320x256', '400x225', '400x250', '400x300', '512x288', '512x320', '512x384',
               '640x360', '640x400', '640x480', '800x450', '800x500', '800x600', '1024x576', '1024x640',
               '1024x768', '1280x720', '1280x800', '1280x960', '1280x1024', '1400x787', '1400x875',
               '1400x1050', '1600x900', '1600x1000', '1600x1200', '1920x1080']

__all__ = [ 'SetResolution' ]

def SetResolution(target_resolution):

    class SetResolutionWrapper(gym.Wrapper):
        """
            Doom wrapper to change screen resolution
        """
        def __init__(self, env):
            super(SetResolutionWrapper, self).__init__(env)
            if target_resolution not in resolutions:
                raise gym.error.Error('Error - The specified resolution "{}" is not supported by Vizdoom.'.format(target_resolution))
            parts = target_resolution.lower().split('x')
            width = int(parts[0])
            height = int(parts[1])
            screen_res = target_resolution
            self.screen_width, self.screen_height, self.unwrapped.screen_resolution = width, height, screen_res
            self.unwrapped.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
            self.observation_space = self.unwrapped.observation_space

    return SetResolutionWrapper

# Adapters
from gym.spaces import Discrete, MultiDiscrete
class DiscreteToMultiDiscrete(Discrete):
    """
    Adapter that adapts the MultiDiscrete action space to a Discrete action space of any size
    The converted action can be retrieved by calling the adapter with the discrete action
        discrete_to_multi_discrete = DiscreteToMultiDiscrete(multi_discrete)
        discrete_action = discrete_to_multi_discrete.sample()
        multi_discrete_action = discrete_to_multi_discrete(discrete_action)
    It can be initialized using 3 configurations:
    Configuration 1) - DiscreteToMultiDiscrete(multi_discrete)                   [2nd param is empty]
        Would adapt to a Discrete action space of size (1 + nb of discrete in MultiDiscrete)
        where
            0   returns NOOP                                [  0,   0,   0, ...]
            1   returns max for the first discrete space    [max,   0,   0, ...]
            2   returns max for the second discrete space   [  0, max,   0, ...]
            etc.
    Configuration 2) - DiscreteToMultiDiscrete(multi_discrete, list_of_discrete) [2nd param is a list]
        Would adapt to a Discrete action space of size (1 + nb of items in list_of_discrete)
        e.g.
        if list_of_discrete = [0, 2]
            0   returns NOOP                                [  0,   0,   0, ...]
            1   returns max for first discrete in list      [max,   0,   0, ...]
            2   returns max for second discrete in list     [  0,   0,  max, ...]
            etc.
    Configuration 3) - DiscreteToMultiDiscrete(multi_discrete, discrete_mapping) [2nd param is a dict]
        Would adapt to a Discrete action space of size (nb_keys in discrete_mapping)
        where discrete_mapping is a dictionnary in the format { discrete_key: multi_discrete_mapping }
        e.g. for the Nintendo Game Controller [ [0,4], [0,1], [0,1] ] a possible mapping might be;
        mapping = {
            0:  [0, 0, 0],  # NOOP
            1:  [1, 0, 0],  # Up
            2:  [3, 0, 0],  # Down
            3:  [2, 0, 0],  # Right
            4:  [2, 1, 0],  # Right + A
            5:  [2, 0, 1],  # Right + B
            6:  [2, 1, 1],  # Right + A + B
            7:  [4, 0, 0],  # Left
            8:  [4, 1, 0],  # Left + A
            9:  [4, 0, 1],  # Left + B
            10: [4, 1, 1],  # Left + A + B
            11: [0, 1, 0],  # A only
            12: [0, 0, 1],  # B only,
            13: [0, 1, 1],  # A + B
        }
    """
    def __init__(self, multi_discrete, options=None):
        assert isinstance(multi_discrete, MultiDiscrete)
        self.multi_discrete = multi_discrete
        self.num_discrete_space = self.multi_discrete.num_discrete_space

        # Config 1
        if options is None:
            self.n = self.num_discrete_space + 1                # +1 for NOOP at beginning
            self.mapping = {i: [0] * self.num_discrete_space for i in range(self.n)}
            for i in range(self.num_discrete_space):
                self.mapping[i + 1][i] = self.multi_discrete.high[i]

        # Config 2
        elif isinstance(options, list):
            assert len(options) <= self.num_discrete_space
            self.n = len(options) + 1                          # +1 for NOOP at beginning
            self.mapping = {i: [0] * self.num_discrete_space for i in range(self.n)}
            for i, disc_num in enumerate(options):
                assert disc_num < self.num_discrete_space
                self.mapping[i + 1][disc_num] = self.multi_discrete.high[disc_num]

        # Config 3
        elif isinstance(options, dict):
            self.n = len(options.keys())
            self.mapping = options
            for i, key in enumerate(options.keys()):
                if i != key:
                    raise Error('DiscreteToMultiDiscrete must contain ordered keys. ' \
                                'Item {0} should have a key of "{0}", but key "{1}" found instead.'.format(i, key))
                if not self.multi_discrete.contains(options[key]):
                    raise Error('DiscreteToMultiDiscrete mapping for key {0} is ' \
                                'not contained in the underlying MultiDiscrete action space. ' \
                                'Invalid mapping: {1}'.format(key, options[key]))
        # Unknown parameter provided
        else:
            raise Error('DiscreteToMultiDiscrete - Invalid parameter provided.')

    def __call__(self, discrete_action):
        return self.mapping[discrete_action]

# Discrete Action Wrapper
# Constants
NUM_ACTIONS = 43
ALLOWED_ACTIONS = [
    [0, 10, 11],                                # 0 - Basic
    [0, 10, 11, 13, 14, 15],                    # 1 - Corridor
    [0, 14, 15],                                # 2 - DefendCenter
    [0, 14, 15],                                # 3 - DefendLine
    [13, 14, 15],                               # 4 - HealthGathering
    [13, 14, 15],                               # 5 - MyWayHome
    [0, 14, 15],                                # 6 - PredictPosition
    [10, 11],                                   # 7 - TakeCover
    [x for x in range(NUM_ACTIONS) if x != 33], # 8 - Deathmatch
    [13, 14, 15],                               # 9 - MyWayHomeFixed
    [13, 14, 15],                               # 10 - MyWayHomeFixed15
    [13, 14, 15],                               # 10 - MyWayHomeFixed15
    [13, 14, 15],                               # 10 - MyWayHomeFixed15
    [13, 14, 15],                               # 10 - MyWayHomeFixed15
    [13, 14, 15],                               # 10 - MyWayHomeFixed15
    [13, 14, 15],                               # 10 - MyWayHomeFixed15
    [13, 14, 15],                               # 10 - MyWayHomeFixed15
    [13, 14, 15]
]


def ToDiscrete(config):
    # Config can be 'minimal', 'constant-7', 'constant-17', 'full'

    class ToDiscreteWrapper(gym.Wrapper):
        """
            Doom wrapper to convert MultiDiscrete action space to Discrete
            config:
                - minimal - Will only use the levels' allowed actions (+ NOOP)
                - constant-7 - Will use the 7 minimum actions (+NOOP) to complete all levels
                - constant-17 - Will use the 17 most common actions (+NOOP) to complete all levels
                - full - Will use all available actions (+ NOOP)
            list of commands:
                - minimal:
                    Basic:              NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT
                    Corridor:           NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
                    DefendCenter        NOOP, ATTACK, TURN_RIGHT, TURN_LEFT
                    DefendLine:         NOOP, ATTACK, TURN_RIGHT, TURN_LEFT
                    HealthGathering:    NOOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
                    MyWayHome:          NOOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
                    PredictPosition:    NOOP, ATTACK, TURN_RIGHT, TURN_LEFT
                    TakeCover:          NOOP, MOVE_RIGHT, MOVE_LEFT
                    Deathmatch:         NOOP, ALL COMMANDS (Deltas are limited to [0,1] range and will not work properly)
                - constant-7: NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, SELECT_NEXT_WEAPON
                - constant-17: NOOP, ATTACK, JUMP, CROUCH, TURN180, RELOAD, SPEED, STRAFE, MOVE_RIGHT, MOVE_LEFT, MOVE_BACKWARD
                                MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, LOOK_UP, LOOK_DOWN, SELECT_NEXT_WEAPON, SELECT_PREV_WEAPON
        """
        def __init__(self, env):
            super(ToDiscreteWrapper, self).__init__(env)
            if config == 'minimal':
                allowed_actions = ALLOWED_ACTIONS[self.unwrapped.level]
            elif config == 'constant-7':
                allowed_actions = [0, 10, 11, 13, 14, 15, 31]
            elif config == 'constant-17':
                allowed_actions = [0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32]
            elif config == 'full':
                allowed_actions = None
            else:
                raise gym.error.Error('Invalid configuration. Valid options are "minimal", "constant-7", "constant-17", "full"')
            self.action_space = DiscreteToMultiDiscrete(self.action_space, allowed_actions)
        def _step(self, action):
            return self.env._step(self.action_space(action))

    return ToDiscreteWrapper

def wrap_vizdoom(env, stack=4):
    # resolution_wrapper = SetResolution("160x120")
    # env = resolution_wrapper(env)
    env = MaxAndSkipEnv(env, skip=4, max_over=1)
    env = WarpFrame(env, res=42)
    # env = ClipNegativeRewardEnv(env)
    env = FrameStack(env, 4)
    env = GreyscaleRender(env)
    discrete_action_wrapper = ToDiscrete("minimal")
    env = discrete_action_wrapper(env)
    print("Wrapping environment with vizdoom settings.")
    return env
