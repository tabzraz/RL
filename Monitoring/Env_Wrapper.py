import numpy as np
from skimage import draw
import gym
import pygame


class EnvWrapper(gym.Env):

    def __init__(self, env, debug, args):
        self.wrapped_env = env
        self.metadata = env.metadata
        self.made_screen = False
        self.debug = debug
        self.scaling = 8
        self.args = args

    def _step(self, a):
        return self.wrapped_env.step(a)

    def _reset(self):
        return self.wrapped_env.reset()

    def _render(self, mode="human", close=False):
        return self.wrapped_env.render(mode, close)

    def visitations(self, player_positions):
        try:
            return self.wrapped_env.env.player_visits(player_positions, self.args)
        except AttributeError:
            print("No visitations for this environment")

    def frontier(self, exp_model):
        try:
            return self.wrapped_env.env.frontier(exp_model, self.args)
        except AttributeError:
            print("No frontier for this environment")

    def explorations(self, player_positions, exploration_bonuses, max_bonus):
        try:
            return self.wrapped_env.env.bonus_landscape(player_positions, exploration_bonuses, max_bonus, self.args)
        except AttributeError:
            print("No bonus landscape for this environment")

    def log_visitation(self):
        try:
            return self.wrapped_env.env.log_player_pos()
        except AttributeError:
            pass

    def debug_render(self, debug_info={}, mode="human", close=False):
        if self.debug:

            if mode == "human":
                if close:
                    pygame.quit()
                    return
                rgb_array = self.debug_render(debug_info, mode="rgb_array")
                if not self.made_screen:
                    pygame.init()
                    screen_size = (rgb_array.shape[1] * self.scaling, rgb_array.shape[0] * self.scaling)
                    screen = pygame.display.set_mode(screen_size)
                    self.screen = screen
                    self.made_screen = True
                self.screen.fill((0, 0, 0))
                for x in range(rgb_array.shape[0]):
                    for y in range(rgb_array.shape[1]):
                        if not np.all(rgb_array[x, y, :] == (0, 0, 0)):
                            pygame.draw.rect(self.screen, rgb_array[x, y], (y * self.scaling, x * self.scaling, self.scaling, self.scaling))
                pygame.display.update()

            elif mode == "rgb_array":
                env_image = self.wrapped_env.render(mode="rgb_array")
                env_image = np.swapaxes(env_image, 0, 1)
                image_x = env_image.shape[0]
                image_y = env_image.shape[1] + 1

                if "CTS_State" in debug_info:
                    image_x += 5 + debug_info["CTS_State"].shape[0]
                if "Q_Values" in debug_info:
                    image_y += 50

                image = np.zeros(shape=(image_x, image_y, 3))
                image[:env_image.shape[0], :env_image.shape[1], :] = env_image

                # Draw the Q-Values
                if "Q_Values" in debug_info:
                    q_vals_image = self.draw_q_values(debug_info, env_image.shape[0] - 1, 48)
                    image[1:env_image.shape[0], env_image.shape[1] + 2:env_image.shape[1] + 50, :] = q_vals_image

                # Draw the Pseudo-Count stuff
                if "CTS_State" in debug_info:
                    count_image = self.draw_count(debug_info, 5 - 1 + debug_info["CTS_State"].shape[0], image_y - 1)
                    image[env_image.shape[0] + 1:, :-1, :] = count_image

                image = np.swapaxes(image, 0, 1)
                return image
        else:
            return self.render(mode, close)

    def draw_q_values(self, info, width, height):
        image = np.zeros((width, height, 3))
        red = (255, 0, 0)
        yellow = (255, 255, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)
        purple = (255, 0, 255)

        q_values = info["Q_Values"]
        max_q_value = info["Max_Q_Value"]
        min_q_value = info["Min_Q_Value"]
        chosen_action = info["Action"]
        epsilon = info["Epsilon"]
        forced_action = -1
        if "Forced_Action" in info:
            forced_action = info["Forced_Action"]

        q_val_sizes = [int((q_val - min_q_value) / (max_q_value - min_q_value) * (height - 4)) + 4 for q_val in q_values]
        greedy_action = np.argmax(q_values)
        actions = len(q_values)
        bar_width = int(width / actions)

        for i, q_size in enumerate(q_val_sizes):
            if i == forced_action:
                q_color = red
            elif i == greedy_action:
                q_color = yellow
            elif i == chosen_action:
                q_color = green
            else:
                q_color = blue
            rect_coords = [(i * bar_width, 4), (i * bar_width, q_size), ((i + 1) * bar_width, q_size), ((i + 1) * bar_width, 4)]
            rect_row = [r[0] for r in rect_coords]
            rect_col = [r[1] for r in rect_coords]
            rect_array_coords = draw.polygon(rect_row, rect_col)
            draw.set_color(image, rect_array_coords, q_color)

        # Epsilon
        bar_width = int(width * epsilon)
        bonus_rect_coords = [(0, 0), (0, 3), (bar_width, 3), (bar_width, 0)]
        rect_row = [r[0] for r in bonus_rect_coords]
        rect_col = [r[1] for r in bonus_rect_coords]
        rect_array_coords = draw.polygon(rect_row, rect_col)
        draw.set_color(image, rect_array_coords, purple)

        return np.fliplr(image)

    def draw_count(self, info, width, height):
        image = np.zeros((width, height, 3))

        red = (255, 0, 0)

        bonus = info["Bonus"]
        max_bonus = info["Max_Bonus"]
        cts_image = info["CTS_State"]
        cts_pg = info["Pixel_PG"]

        # Bar
        bar_height = int(height * bonus / max_bonus)
        bonus_rect_coords = [(0, 0), (0, bar_height), (3, bar_height), (3, 0)]
        rect_row = [r[0] for r in bonus_rect_coords]
        rect_col = [r[1] for r in bonus_rect_coords]
        rect_array_coords = draw.polygon(rect_row, rect_col)
        draw.set_color(image, rect_array_coords, red)

        # PG per pixel
        cts_gray = np.concatenate([cts_image for _ in range(3)], axis=2)
        cts_pg_image = np.empty_like(cts_gray)
        for x in range(cts_image.shape[0]):
            for y in range(cts_image.shape[1]):
                pg = cts_pg[x, y]
                if pg < 0:
                    # Blue
                    pg = max(-1, pg)
                    cts_pg_image[x, y, :] = (0, 0, int(-pg * 255))
                else:
                    # Red
                    pg = min(pg, 1)
                    cts_pg_image[x, y, :] = (int(pg * 255), 0, 0)
        cts_alpha = np.stack([np.abs(np.clip(cts_pg, -1, 1)) for _ in range(3)], axis=2)
        cts_colour_image = cts_alpha * cts_pg_image + (1 - cts_alpha) * cts_gray
        image[4:, -cts_image.shape[1]:, :] = np.fliplr(np.swapaxes(cts_colour_image, 0, 1))

        return np.fliplr(image)

