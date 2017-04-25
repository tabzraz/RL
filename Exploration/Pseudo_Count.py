from .CTS import DensityModel, L_shaped_context
import os
from scipy.misc import imresize as resize
import pickle
import numpy as np
from math import sqrt
from copy import deepcopy
from multiprocessing import Pool


class PseudoCount:

    def __init__(self, args):

        self.beta = args.beta
        self.args = args

        self.cts_model_shape = (args.cts_size, args.cts_size)
        print("\nCTS Model has size: " + str(self.cts_model_shape) + "\n")

        self.cts_model = DensityModel(frame_shape=self.cts_model_shape, context_functor=L_shaped_context, conv=args.cts_conv)
        os.makedirs("{}/cts_model".format(args.log_path))

    def bonus(self, state, dont_remember=False):
        extra_info = {}

        if state.shape[2] != 1:
            raise Exception("Need to grayscale the input to the CTS model")
        state_resized = resize(state[:, :, 0], self.cts_model_shape, mode="F")

        extra_info["CTS_State"] = state_resized[:, :, np.newaxis] * 255

        if dont_remember:
            old_cts_model = deepcopy(self.cts_model)

        rho_old, rho_old_pixels = self.cts_model.update(state_resized)

        rho_new, rho_new_pixels = self.cts_model.log_prob(state_resized)

        if dont_remember:
            self.cts_model = old_cts_model

        pg = rho_new - rho_old

        pg_pixel = rho_new_pixels - rho_old_pixels

        extra_info["Pixel_PG"] = pg_pixel

        pg = min(10, pg)
        pg = max(0.001, pg)
        pseudo_count = 1 / (np.expm1(pg))

        bonus = self.beta / sqrt(pseudo_count + 0.01)
        extra_info["Bonus"] = bonus

        return bonus, extra_info

    def save_model(self):
        return
        # with open("{}/cts_model/cts_model_end.pkl".format(self.args.log_path), "wb") as file:
                # pickle.dump(self.cts_model, file, pickle.HIGHEST_PROTOCOL)
