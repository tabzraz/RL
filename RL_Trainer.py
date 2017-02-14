from __future__ import division, print_function
import argparse
import gym
import time
import datetime
import os
import json
import Envs

parser = argparse.ArgumentParser(description="RL Agent Trainer")
parser.add_argument("--tmax", type=int, default=1e6)
parser.add_argument("--env", type=str, default="Maze-2-v1")
args = parser.parse_args()
print(args)

# FLAGS = flags.FLAGS
# ENV_NAME = FLAGS.env
# RENDER = FLAGS.render
# SLOW = FLAGS.slow
# env = gym.make(ENV_NAME)

# if FLAGS.action_override > 0:
#     ACTIONS = FLAGS.action_override
# else:
#     ACTIONS = env.action_space.n
# DOUBLE_DQN = FLAGS.double
# SEED = FLAGS.seed
# LR = FLAGS.lr
# VIME_LR = FLAGS.vime_lr
# ETA = FLAGS.eta
# VIME_BATCH_SIZE = FLAGS.vime_batch
# VIME_ITERS = FLAGS.vime_iters
# NAME = FLAGS.name
# EPISODES = FLAGS.episodes
# T_MAX = FLAGS.t_max
# EPSILON_START = FLAGS.epsilon_start
# EPSILON_FINISH = FLAGS.epsilon_finish
# EPSILON_STEPS = FLAGS.epsilon_steps
# XP_SIZE = FLAGS.xp
# GAMMA = FLAGS.gamma
# BATCH_SIZE = FLAGS.batch
# TARGET_UPDATE = FLAGS.target
# SUMMARY_UPDATE = FLAGS.summary
# CLIP_VALUE = FLAGS.grad_clip
# VIME = FLAGS.vime
# VIME_POSTERIOR_ITERS = FLAGS.posterior_iters
# RANDOM_AGENT = FLAGS.rnd
# CHECKPOINT_INTERVAL = FLAGS.ckpt_interval
# if FLAGS.ckpt != "":
#     RESTORE = True
#     CHECKPOINT = FLAGS.ckpt
# else:
#     RESTORE = False
# EXPLORATION_STEPS = FLAGS.exp_steps
# if FLAGS.logdir != "":
#     LOGDIR = FLAGS.logdir
# else:
#     LOGDIR = "Logs/{}_{}".format(NAME, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
# DQN_MODEL = FLAGS.model
# LEVELS = FLAGS.levels

# if not os.path.exists("{}/ckpts/".format(LOGDIR)):
#     os.makedirs("{}/ckpts".format(LOGDIR))

# # Print all the hyperparams
# hyperparams = FLAGS.__dict__["__flags"]
# with open("{}/info.json".format(LOGDIR), "w") as fp:
#     json.dump(hyperparams, fp)

# # TODO: Add some more info here
# print("\n--------Info--------")
# print("Logdir:", LOGDIR)
# print("T: {:,}".format(T_MAX))
# print("Actions:", ACTIONS)
# print("Gamma", GAMMA)
# print("Learning Rate:", LR)
# print("Batch Size:", BATCH_SIZE)
# print("VIME bonuses:", VIME)
# print("--------------------\n")

# replays = [ExperienceReplay(XP_SIZE) for _ in range(LEVELS)]

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# with tf.Graph().as_default():
#     with tf.Session(config=config) as sess:

#         # Seed numpy and tensorflow
#         np.random.seed(SEED)
#         tf.set_random_seed(SEED)

#         print("\n-------Models-------")

#         dqn_creator = get_models(DQN_MODEL)
#         dqn = dqn_creator("DQN")
#         target_dqn = dqn_creator("Target_DQN")
