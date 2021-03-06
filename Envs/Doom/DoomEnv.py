import logging
import os
from time import sleep
import multiprocessing

import numpy as np

import gym
from gym import spaces, error
from gym.utils import seeding

# try:
#     import doom_py
#     from doom_py import DoomGame, Mode, Button, GameVariable, ScreenFormat, ScreenResolution, Loader
#     from doom_py.vizdoom import ViZDoomUnexpectedExitException, ViZDoomErrorException
# except ImportError as e:
#     raise gym.error.DependencyNotInstalled("{}. (HINT: you can install Doom dependencies " +
#                                            "with 'pip install doom_py.)'".format(e))

from vizdoom import *

logger = logging.getLogger(__name__)

# Constants
NUM_ACTIONS = 43
NUM_LEVELS = 9
CONFIG = 0
SCENARIO = 1
MAP = 2
DIFFICULTY = 3
ACTIONS = 4
MIN_SCORE = 5
TARGET_SCORE = 6

# Format (config, scenario, map, difficulty, actions, min, target)
DOOM_SETTINGS = [
    ['basic.cfg', 'basic.wad', 'map01', 5, [0, 10, 11], -485, 10],                               # 0 - Basic
    ['deadly_corridor.cfg', 'deadly_corridor.wad', '', 1, [0, 10, 11, 13, 14, 15], -120, 1000],  # 1 - Corridor
    ['defend_the_center.cfg', 'defend_the_center.wad', '', 5, [0, 14, 15], -1, 10],              # 2 - DefendCenter
    ['defend_the_line.cfg', 'defend_the_line.wad', '', 5, [0, 14, 15], -1, 15],                  # 3 - DefendLine
    ['health_gathering.cfg', 'health_gathering.wad', 'map01', 5, [13, 14, 15], 0, 1000],         # 4 - HealthGathering
    ['my_way_home.cfg', 'my_way_home.wad', '', 5, [13, 14, 15], -0.22, 0.5],                     # 5 - MyWayHome
    ['predict_position.cfg', 'predict_position.wad', 'map01', 3, [0, 14, 15], -0.075, 0.5],      # 6 - PredictPosition
    ['take_cover.cfg', 'take_cover.wad', 'map01', 5, [10, 11], 0, 750],                          # 7 - TakeCover
    ['deathmatch.cfg', 'deathmatch.wad', '', 5, [x for x in range(NUM_ACTIONS) if x != 33], 0, 20],  # 8 - Deathmatch
    ['my_way_home.cfg', 'my_way_home_sparse.wad', '', 5, [13, 14, 15], -10, 0.5],                     # 9 - MyWayHome
    ['my_way_home.cfg', 'my_way_home_verySparse.wad', '', 5, [13, 14, 15], -10, 0.5], # 10
    ['maze.cfg', 'maze_1.wad', '', 5, [13, 14, 15], -0.22, 0.5],                     # 11
    ['maze.cfg', 'maze_2.wad', '', 5, [13, 14, 15], -0.22, 0.5],                     # 11
    ['maze.cfg', 'maze_3.wad', '', 5, [13, 14, 15], -0.22, 0.5],                     # 11
    ['maze.cfg', 'maze_4.wad', '', 5, [13, 14, 15], -0.22, 0.5],                     # 11
    ['maze.cfg', 'maze_5.wad', '', 5, [13, 14, 15], -0.22, 0.5]                     # 12
]

# Singleton pattern
class DoomLock:
    class __DoomLock:
        def __init__(self):
            self.lock = multiprocessing.Lock()
    instance = None
    def __init__(self):
        if not DoomLock.instance:
            DoomLock.instance = DoomLock.__DoomLock()
    def get_lock(self):
        return DoomLock.instance.lock


class DoomEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 35}

    def __init__(self, level):
        self.previous_level = -1
        self.level = level
        self.game = DoomGame()
        # self.loader = Loader()
        self.doom_dir = os.path.dirname(os.path.abspath(__file__))
        if level > 8:
            self.doom_dir = "{}".format(os.path.dirname(os.path.realpath(__file__)))
        print("Doom Directory: {}".format(self.doom_dir))
        self._mode = 'algo'                         # 'algo' or 'human'
        self.no_render = False                      # To disable double rendering in human mode
        self.viewer = None
        self.is_initialized = False                 # Indicates that reset() has been called
        self.curr_seed = 0
        self.lock = (DoomLock()).get_lock()
        self.action_space = spaces.MultiDiscrete([[0, 1]] * 38 + [[-10, 10]] * 2 + [[-100, 100]] * 3)
        self.allowed_actions = list(range(NUM_ACTIONS))
        self.screen_height = 480
        self.screen_width = 640
        self.screen_resolution = ScreenResolution.RES_160X120
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))

        # Vis stuff
        self.x = 0
        self.y = 0

        self._seed()
        self._configure()

    def _configure(self, lock=None, **kwargs):
        if 'screen_resolution' in kwargs:
            logger.warn('Deprecated - Screen resolution must now be set using a wrapper. See documentation for details.')
        # Multiprocessing lock
        if lock is not None:
            self.lock = lock

    def _load_level(self):
        # Closing if is_initialized
        if self.is_initialized:
            self.is_initialized = False
            self.game.close()
            self.game = DoomGame()

        # Customizing level
        if getattr(self, '_customize_game', None) is not None and callable(self._customize_game):
            self.level = -1
            self._customize_game()

        else:
            # Loading Paths
            # if not self.is_initialized:
            #     self.game.set_vizdoom_path(self.loader.get_vizdoom_path())
            #     self.game.set_doom_game_path(self.loader.get_freedoom_path())

            # Common settings
            self.game.load_config(os.path.join(self.doom_dir, 'assets/%s' % DOOM_SETTINGS[self.level][CONFIG]))
            # self.game.set_doom_scenario_path(self.loader.get_scenario_path(DOOM_SETTINGS[self.level][SCENARIO]))
            if self.level > 8:
                # print("Setting scenario path")
                self.game.set_doom_scenario_path("{}/assets/{}".format(self.doom_dir, DOOM_SETTINGS[self.level][SCENARIO]))
            if DOOM_SETTINGS[self.level][MAP] != '':
                self.game.set_doom_map(DOOM_SETTINGS[self.level][MAP])
            self.game.set_doom_skill(DOOM_SETTINGS[self.level][DIFFICULTY])
            self.allowed_actions = DOOM_SETTINGS[self.level][ACTIONS]
            self.game.set_screen_resolution(self.screen_resolution)
            self.game.clear_available_game_variables()
            self.game.add_available_game_variable(GameVariable.POSITION_X)
            self.game.add_available_game_variable(GameVariable.POSITION_Y)

        self.previous_level = self.level
        self._closed = False

        # self.game.clear_game_args()
        # self.game.add_game_args('+viz_nocheat 0')
        # Algo mode
        if 'human' != self._mode:
            # We want the extra information for visualisations
            self.game.set_window_visible(False)
            self.game.set_mode(Mode.PLAYER)
            self.no_render = False
            try:
                with self.lock:
                    self.game.init()
            except (ViZDoomUnexpectedExitException, ViZDoomErrorException):
                raise error.Error(
                    'VizDoom exited unexpectedly. This is likely caused by a missing multiprocessing lock. ' +
                    'To run VizDoom across multiple processes, you need to pass a lock when you configure the env ' +
                    '[e.g. env.configure(lock=my_multiprocessing_lock)], or create and close an env ' +
                    'before starting your processes [e.g. env = gym.make("DoomBasic-v0"); env.close()] to cache a ' +
                    'singleton lock in memory.')
            self._start_episode()
            self.is_initialized = True
            return self.game.get_state().screen_buffer.copy()

        # Human mode
        else:
            self.game.add_game_args('+freelook 1')
            self.game.set_window_visible(True)
            self.game.set_mode(Mode.SPECTATOR)
            self.no_render = True
            with self.lock:
                self.game.init()
            self._start_episode()
            self.is_initialized = True
            self._play_human_mode()
            return np.zeros(shape=self.observation_space.shape, dtype=np.uint8)

    def _start_episode(self):
        if self.curr_seed > 0:
            self.game.set_seed(self.curr_seed)
            self.curr_seed = 0
        self.game.new_episode()
        return

    def _play_human_mode(self):
        while not self.game.is_episode_finished():
            self.game.advance_action()
            state = self.game.get_state()
            total_reward = self.game.get_total_reward()
            info = self._get_game_variables(state.game_variables)
            info["TOTAL_REWARD"] = round(total_reward, 4)
            print('===============================')
            print('State: #' + str(state.number))
            print('Action: \t' + str(self.game.get_last_action()) + '\t (=> only allowed actions)')
            print('Reward: \t' + str(self.game.get_last_reward()))
            print('Total Reward: \t' + str(total_reward))
            print('Variables: \n' + str(info))
            sleep(0.02857)  # 35 fps = 0.02857 sleep between frames
        print('===============================')
        print('Done')
        return

    def _step(self, action):
        if NUM_ACTIONS != len(action):
            logger.warn('Doom action list must contain %d items. Padding missing items with 0' % NUM_ACTIONS)
            old_action = action
            action = [0] * NUM_ACTIONS
            for i in range(len(old_action)):
                action[i] = old_action[i]
        # action is a list of numbers but DoomGame.make_action expects a list of ints
        if len(self.allowed_actions) > 0:
            list_action = [int(action[action_idx]) for action_idx in self.allowed_actions]
        else:
            list_action = [int(x) for x in action]

        reward = self.game.make_action(list_action)
        state = self.game.get_state()
        info = {}
        if not self.game.is_episode_finished():
            # info = self._get_game_variables(state.game_variables)
            # print(state.game_variables)
            self.x = state.game_variables[0]
            self.y = state.game_variables[1]
            # info["TOTAL_REWARD"] = round(self.game.get_total_reward(), 4)

        if self.game.is_episode_finished():
            is_finished = True
            if reward < 0.1:
                # The environment ended because we took the time_limit number of timesteps
                info["Steps_Termination"] = True
            # print("Finished steps", reward, is_finished)
            return np.zeros(shape=self.observation_space.shape, dtype=np.uint8), reward, is_finished, info
        else:
            is_finished = False
            # print("Finished", reward, is_finished)
            return state.screen_buffer.copy(), reward, is_finished, info

    def _reset(self):
        if self.is_initialized and not self._closed:
            self._start_episode()
            image_buffer = self.game.get_state().screen_buffer
            if image_buffer is None:
                raise error.Error(
                    'VizDoom incorrectly initiated. This is likely caused by a missing multiprocessing lock. ' +
                    'To run VizDoom across multiple processes, you need to pass a lock when you configure the env ' +
                    '[e.g. env.configure(lock=my_multiprocessing_lock)], or create and close an env ' +
                    'before starting your processes [e.g. env = gym.make("DoomBasic-v0"); env.close()] to cache a ' +
                    'singleton lock in memory.')
            return image_buffer.copy()
        else:
            return self._load_level()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None      # If we don't None out this reference pyglet becomes unhappy
            return
        try:
            if 'human' == mode and self.no_render:
                return
            state = self.game.get_state()
            img = state.screen_buffer
            # VizDoom returns None if the episode is finished, let's make it
            # an empty image so the recorder doesn't stop
            if img is None:
                img = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
            if mode == 'rgb_array':
                return img
            elif mode is 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
        except doom_py.vizdoom.ViZDoomIsNotRunningException:
            pass  # Doom has been closed

    def _close(self):
        # Lock required for VizDoom to close processes properly
        with self.lock:
            self.game.close()

    def _seed(self, seed=None):
        self.curr_seed = seeding.hash_seed(seed) % 2 ** 32
        return [self.curr_seed]

    def _get_game_variables(self, state_variables):
        info = {
            "LEVEL": self.level
        }
        if state_variables is None:
            return info
        # info['KILLCOUNT'] = state_variables[0]
        # info['ITEMCOUNT'] = state_variables[1]
        # info['SECRETCOUNT'] = state_variables[2]
        # info['FRAGCOUNT'] = state_variables[3]
        # info['HEALTH'] = state_variables[4]
        # info['ARMOR'] = state_variables[5]
        # info['DEAD'] = state_variables[6]
        # info['ON_GROUND'] = state_variables[7]
        # info['ATTACK_READY'] = state_variables[8]
        # info['ALTATTACK_READY'] = state_variables[9]
        # info['SELECTED_WEAPON'] = state_variables[10]
        # info['SELECTED_WEAPON_AMMO'] = state_variables[11]
        # info['AMMO1'] = state_variables[12]
        # info['AMMO2'] = state_variables[13]
        # info['AMMO3'] = state_variables[14]
        # info['AMMO4'] = state_variables[15]
    # info['AMMO5'] = state_variables[16]
        # info['AMMO6'] = state_variables[17]
        # info['AMMO7'] = state_variables[18]
        # info['AMMO8'] = state_variables[19]
        # info['AMMO9'] = state_variables[20]
        # info['AMMO0'] = state_variables[21]
        return info

    def log_player_pos(self):
        return(int(self.x / 10), int(self.y / 10))

    # At the moment this is just to save the player positions
    def player_visits(self, player_visits, args):
        # Log the visitations
        with open("{}/logs/Player_Positions.txt".format(args.log_path), "a") as file:
            file.write('\n'.join(" ".join(str(x) for x in t) for t in player_visits))

        return None
