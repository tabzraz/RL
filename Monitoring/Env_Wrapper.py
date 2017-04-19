import numpy as np
from skimage import draw
import gym
import pygame


class EnvWrapper(gym.Env):

    def __init__(self, env):
        self.wrapped_env = env
        self.metadata = env.metadata
        self.made_screen = False

    def _step(self, a):
        return self.wrapped_env.step(a)

    def _reset(self):
        return self.wrapped_env.reset()

    def _render(self, mode="human", close=False):
        return self.wrapped_env.render(mode, close)

    def debug_render(self, debug, debug_info={}, mode="human", close=False):
        if debug:

            if mode == "human":
                rgb_array = self.debug_render(debug, debug_info, mode="rgb_array")
                if not self.made_screen:
                    pygame.init()
                    screen_size = (rgb_array.shape[0] * 4, rgb_array.shape[1] * 4)
                    screen = pygame.display.set_mode(screen_size)
                    self.screen = screen
                    self.made_screen = True
                self.screen.fill((0, 0, 0))
                for x in range(rgb_array.shape[0]):
                    for y in range(rgb_array.shape[1]):
                        if rgb_array[x, y] != (0, 0, 0):
                            pygame.draw.rect(self.screen, rgb_array, (x * 4, y * 4, 4, 4))
                pygame.display.update()

            elif mode == "rgb_array":
                env_image = self.wrapped_env.render(mode="rgb_array")
                image_x = env_image.shape[0]
                image_y = env_image.shape[1]

                if "CTS_State" in debug_info:
                    image_x += 10 + debug_info["CTS_State"].shape[0]
                if "Q_Values" in debug_info:
                    image_y += 50

                image = np.zeros(shape=(image_x, image_y, 3))
                image[:env_image.shape[0], :env_image.shape[1], :] = env_image

                return image
        else:
            self.render(mode, close)
