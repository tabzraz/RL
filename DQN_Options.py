from __future__ import division, print_function
import numpy as np
import gym
import tensorflow as tf
import time
import datetime
import os
import json
from Misc.Gradients import clip_grads
from Replay.ExpReplay_Options import ExperienceReplay_Options as ExperienceReplay
from Models.Models import get_models
from Misc.Utils import time_str
import Envs

flags = tf.app.flags
flags.DEFINE_string("env", "Maze-4-v0", "Environment name for OpenAI gym")
flags.DEFINE_string("logdir", "", "Directory to put logs (including tensorboard logs)")
flags.DEFINE_string("name", "nn", "The name of the model")
flags.DEFINE_float("lr", 0.0001, "Initial Learning Rate")
flags.DEFINE_float("vime_lr", 0.0001, "Initial Learning Rate for VIME model")
flags.DEFINE_float("gamma", 0.99, "Gamma, the discount rate for future rewards")
flags.DEFINE_integer("t_max", int(2e5), "Number of frames to act for")
flags.DEFINE_integer("episodes", 100, "Number of episodes to act for")
flags.DEFINE_integer("action_override", 0, "Overrides the number of actions provided by the environment")
flags.DEFINE_float("grad_clip", 50, "Clips gradients by their norm")
flags.DEFINE_integer("seed", 0, "Seed for numpy and tf")
flags.DEFINE_integer("ckpt_interval", 2e4, "How often to save the global model")
flags.DEFINE_integer("xp", int(5e4), "Size of the experience replay")
flags.DEFINE_float("epsilon_start", 1.0, "Value of epsilon to start with")
flags.DEFINE_float("epsilon_finish", 0.01, "Final value of epsilon to anneal to")
flags.DEFINE_integer("epsilon_steps", int(15e4), "Number of steps to anneal epsilon for")
flags.DEFINE_integer("target", 500, "After how many steps to update the target network")
flags.DEFINE_boolean("double", True, "Double DQN or not")
flags.DEFINE_integer("batch", 64, "Minibatch size")
flags.DEFINE_integer("summary", 10, "After how many steps to log summary info")
flags.DEFINE_integer("exp_steps", int(5e4), "Number of steps to randomly explore for")
flags.DEFINE_boolean("render", False, "Render environment or not")
flags.DEFINE_string("ckpt", "", "Model checkpoint to restore")
flags.DEFINE_boolean("vime", False, "Whether to add VIME style exploration bonuses")
flags.DEFINE_integer("posterior_iters", 1, "Number of times to run gradient descent for calculating new posterior from old posterior")
flags.DEFINE_float("eta", 1.0, "How much to scale the VIME rewards")
flags.DEFINE_integer("vime_iters", 50, "Number of iterations to optimise the variational baseline for VIME")
flags.DEFINE_integer("vime_batch", 32, "Size of minibatch for VIME")
flags.DEFINE_boolean("rnd", False, "Random Agent")
flags.DEFINE_boolean("slow", False, "Slow down the loop")
flags.DEFINE_string("model", "Maze-1", "The name of the model to use for the dqn")
flags.DEFINE_integer("levels", 1, "Number of levels in the hierarchy")

FLAGS = flags.FLAGS
ENV_NAME = FLAGS.env
RENDER = FLAGS.render
SLOW = FLAGS.slow
env = gym.make(ENV_NAME)

if FLAGS.action_override > 0:
    ACTIONS = FLAGS.action_override
else:
    ACTIONS = env.action_space.n
DOUBLE_DQN = FLAGS.double
SEED = FLAGS.seed
LR = FLAGS.lr
VIME_LR = FLAGS.vime_lr
ETA = FLAGS.eta
VIME_BATCH_SIZE = FLAGS.vime_batch
VIME_ITERS = FLAGS.vime_iters
NAME = FLAGS.name
EPISODES = FLAGS.episodes
T_MAX = FLAGS.t_max
EPSILON_START = FLAGS.epsilon_start
EPSILON_FINISH = FLAGS.epsilon_finish
EPSILON_STEPS = FLAGS.epsilon_steps
XP_SIZE = FLAGS.xp
GAMMA = FLAGS.gamma
BATCH_SIZE = FLAGS.batch
TARGET_UPDATE = FLAGS.target
SUMMARY_UPDATE = FLAGS.summary
CLIP_VALUE = FLAGS.grad_clip
VIME = FLAGS.vime
VIME_POSTERIOR_ITERS = FLAGS.posterior_iters
RANDOM_AGENT = FLAGS.rnd
CHECKPOINT_INTERVAL = FLAGS.ckpt_interval
if FLAGS.ckpt != "":
    RESTORE = True
    CHECKPOINT = FLAGS.ckpt
else:
    RESTORE = False
EXPLORATION_STEPS = FLAGS.exp_steps
if FLAGS.logdir != "":
    LOGDIR = FLAGS.logdir
else:
    LOGDIR = "Logs/{}_{}".format(NAME, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
DQN_MODEL = FLAGS.model
LEVELS = FLAGS.levels

if not os.path.exists("{}/ckpts/".format(LOGDIR)):
    os.makedirs("{}/ckpts".format(LOGDIR))

# Print all the hyperparams
hyperparams = FLAGS.__dict__["__flags"]
with open("{}/info.json".format(LOGDIR), "w") as fp:
    json.dump(hyperparams, fp)

# TODO: Add some more info here
print("\n--------Info--------")
print("Logdir:", LOGDIR)
print("T: {:,}".format(T_MAX))
print("Actions:", ACTIONS)
print("Gamma", GAMMA)
print("Learning Rate:", LR)
print("Batch Size:", BATCH_SIZE)
print("VIME bonuses:", VIME)
print("--------------------\n")

replays = [ExperienceReplay(XP_SIZE) for _ in range(LEVELS)]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Graph().as_default():
    with tf.Session(config=config) as sess:

        # Seed numpy and tensorflow
        np.random.seed(SEED)
        tf.set_random_seed(SEED)

        print("\n-------Models-------")

        dqn_creator = get_models(DQN_MODEL)
        dqn = dqn_creator("DQN")
        target_dqn = dqn_creator("Target_DQN")
