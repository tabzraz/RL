# from .CTS import DensityModel, L_shaped_context
from .CTS import TreeDensity
import os
from scipy.misc import imresize as resize
import pickle
import numpy as np
from math import sqrt


class PseudoCount:

    def __init__(self, args):

        self.beta = args.beta
        self.args = args

        self.cts_model_shape = (args.cts_size, args.cts_size)
        print("\nCTS Model has size: " + str(self.cts_model_shape) + "\n")

        # self.cts_model = DensityModel(frame_shape=self.cts_model_shape, context_functor=L_shaped_context, conv=args.cts_conv)
        self.actions = 1
        if self.args.count_state_action:
            self.actions = self.args.actions
        self.models = [TreeDensity(frame_shape=self.cts_model_shape) for _ in range(self.actions)]

        os.makedirs("{}/exploration_model".format(args.log_path))

    def bonus(self, state, action=0, dont_remember=False):
        extra_info = {}

        if state.shape[2] != 1:
            raise Exception("Need to grayscale the input to the CTS model")
        state_resized = resize(state[:, :, 0], self.cts_model_shape, mode="F", interp="bilinear")

        extra_info["CTS_State"] = state_resized[:, :, np.newaxis] * 255

        # if dont_remember:
            # old_cts_model = deepcopy(self.cts_model)

        # rho_old, rho_old_pixels = self.cts_model.update(state_resized)

        # rho_new, rho_new_pixels = self.cts_model.log_prob(state_resized)

        # if dont_remember:
            # self.cts_model = old_cts_model

        pg, pg_pixel, density = self.models[action].new_old(state_resized, keep=not dont_remember)

        # pg = rho_new - rho_old

        # pg_pixel = rho_new_pixels - rho_old_pixels

        extra_info["Pixel_PG"] = pg_pixel

        # print(density)
        density = min(1, density)
        density = max(density, 1e-50)
        extra_info["Density"] = density

        # Don't want prediction gain to be too large or too small
        pg = min(100, pg)
        pg = max(0.00001, pg)
        pseudo_count = 1 / (np.expm1(pg))
        extra_info["Pseudo_Count"] = pseudo_count

        bonus = self.beta / sqrt(pseudo_count + 0.01)

        if self.args.bonus_clip >= 0 and bonus < self.beta * self.args.bonus_clip:
            bonus = 0
        if self.args.negative_rewards:
            bonus -= self.beta * self.args.negative_reward_threshold

        extra_info["Bonus"] = bonus

        return bonus, extra_info

    def save_model(self):
        pass
        # The tree model takes up TOO MUCH space to store (~800mb!)
        # with open("{}/exploration_model/model_end.pkl".format(self.args.log_path), "wb") as file:
            # pickle.dump(self.model, file, pickle.HIGHEST_PROTOCOL)
