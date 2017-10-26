import argparse
import datetime
import gym
import os
import time
import tarfile
import shutil
import Envs
from Utils.Utils import time_str
from RL_Trainer_PyTorch import Trainer
from Envs.OpenAI_AtariWrapper import wrap_deepmind, wrap_vizdoom, wrap_maze

# Argument passing stuff is here
# TODO: Spread these out into useful groups and provide comments
parser = argparse.ArgumentParser(description="RL Agent Trainer")
parser.add_argument("--t-max", type=int, default=int(1e3))
parser.add_argument("--env", type=str, default="Room-14-v0")
parser.add_argument("--logdir", type=str, default="Logs")
parser.add_argument("--name", type=str, default="nn")
parser.add_argument("--exp-replay-size", "--xp", type=int, default=int(1e4))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--model", type=str, default="")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--render", action="store_true", default=False)
parser.add_argument("--slow-render", action="store_true", default=False)
parser.add_argument("--epsilon-steps", "--eps-steps", type=int, default=int(60e3))
parser.add_argument("--epsilon-start", "--eps-start", type=float, default=1.0)
parser.add_argument("--epsilon-finish", "--eps-finish", type=float, default=0.1)
parser.add_argument("--batch-size", "--batch", type=int, default=32)
parser.add_argument("--iters", type=int, default=1)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--target", type=int, default=500)
parser.add_argument("--count", action="store_true", default=False)
parser.add_argument("--beta", type=float, default=0.01)
parser.add_argument("--exploration-steps", "--exp-steps", type=int, default=int(50e1))
parser.add_argument("--double", action="store_true", default=False)
parser.add_argument("--n-step", "--n", type=int, default=1)
parser.add_argument("--n-inc", action="store_true", default=False)
parser.add_argument("--n-max", type=int, default=10)
parser.add_argument("--plain-print", action="store_true", default=False)
parser.add_argument("--clip-value", type=float, default=5)
parser.add_argument("--no-tb", action="store_true", default=False)
parser.add_argument("--no-eval-images", action="store_true", default=False)
parser.add_argument("--eval-images-interval", type=int, default=4)
parser.add_argument("--tb-interval", type=int, default=10)
parser.add_argument("--debug-eval", action="store_true", default=False)
parser.add_argument("--cts-size", type=int, default=2)
parser.add_argument("--cts-conv", action="store_true", default=False)
parser.add_argument("--exp-bonus-save", type=float, default=1.1)
parser.add_argument("--clip-reward", action="store_true", default=False)
parser.add_argument("--options", type=str, default="Primitive")
parser.add_argument("--num-macros", type=int, default=10)
parser.add_argument("--max-macro-length", type=int, default=10)
parser.add_argument("--macro-seed", type=int, default=12)
parser.add_argument("--train-primitives", action="store_true", default=False)
parser.add_argument("--no-visitations", action="store_true", default=False)
parser.add_argument("--interval-size", type=int, default=100)
parser.add_argument("--stale-limit", type=int, default=int(1e6))
parser.add_argument("--count-epsilon", action="store_true", default=False)
parser.add_argument("--epsilon-scaler", type=float, default=1)
parser.add_argument("--tar", action="store_true", default=False)
parser.add_argument("--no-frontier", action="store_true", default=False)
parser.add_argument("--frontier-interval", type=int, default=100)
parser.add_argument("--count-state-action", action="store_true", default=False)
parser.add_argument("--force-low-count-action", action="store_true", default=False)
parser.add_argument("--min-action-count", type=int, default=10)
parser.add_argument("--prioritized", "--priority", action="store_true", default=False)
parser.add_argument("--prioritized-is", action="store_true", default=False)
parser.add_argument("--no-exploration-bonus", "--no-exp-bonus", action="store_true", default=False)
parser.add_argument("--epsilon-decay", action="store_true", default=False)
parser.add_argument("--decay-rate", type=float, default=0.99)
parser.add_argument("--negative-rewards", action="store_true", default=False)
parser.add_argument("--negative-reward-threshold", type=float, default=0.2)
parser.add_argument("--eligibility-trace", "--et", action="store_true", default=False)
parser.add_argument("--lambda_", type=float, default=0.8)
parser.add_argument("--num-states", type=int, default=5)
parser.add_argument("--gap", type=int, default=2)
parser.add_argument("--count-td-priority", action="store_true", default=False)
parser.add_argument("--n-step-mixing", type=float, default=1.0)
parser.add_argument("--set-replay", action="store_true", default=False)
parser.add_argument("--set-replay-num", type=int, default=1)
parser.add_argument("--bonus-clip", type=float, default=-1.0)
parser.add_argument("--tabular", action="store_true", default=False)
parser.add_argument("--log-trained-on-states", type=bool, default=True)
parser.add_argument("--count-td-scaler", type=float, default=1)
parser.add_argument("--variable-n-step", action="store_true", default=False)
parser.add_argument("--negative-td-scaler", type=float, default=1)
parser.add_argument("--density-priority", action="store_true", default=False)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--render-scaling", type=int, default=8)
parser.add_argument("--eval-interval", type=int, default=100)
parser.add_argument("--ucb", action="store_true", default=False)

parser.add_argument("--optimistic-init", action="store_true", default=False)
parser.add_argument("--optimistic-scaler", type=float, default=10)
parser.add_argument("--bandit-p", type=float, default=(1/2))

parser.add_argument("--goal-dqn", action="store_true", default=False)
parser.add_argument("--goal-state-interval", type=int, default=1000)
parser.add_argument("--goal-state-threshold", type=float, default=0.5)
parser.add_argument("--goal-iters", type=int, default=100)
parser.add_argument("--max-option-steps", type=int, default=100)

parser.add_argument("--bonus-replay", action="store_true", default=False)
parser.add_argument("--bonus-replay-size", type=int, default=int(1e4))
parser.add_argument("--bonus-replay-threshold", type=float, default=0.75)

parser.add_argument("--sarsa", action="store_true", default=False)
parser.add_argument("--sarsa-train", type=int, default=100)

parser.add_argument("--nec", action="store_true", default=False)
parser.add_argument("--dnd-size", type=int, default=100)
parser.add_argument("--nec-embedding", type=int, default=2)
parser.add_argument("--nec-alpha", type=float, default=0.1)
parser.add_argument("--nec-neighbours", type=int, default=10)
parser.add_argument("--nec-update", type=int, default=10)

parser.add_argument("--one-step-bonus", action="store_true", default=False)

parser.add_argument("--distrib", action="store_true", default=False)
parser.add_argument("--v-min", type=int, default=-1)
parser.add_argument("--v-max", type=int, default=1)
parser.add_argument("--atoms", type=int, default=2)

parser.add_argument("--model-dqn", action="store_true", default=False)
parser.add_argument("--model-loss", type=float, default=0.25)
parser.add_argument("--model-save-image", type=int, default=10000)
parser.add_argument("--lookahead-plan", action="store_true", default=False)
parser.add_argument("--lookahead-depth", type=int, default=1)
parser.add_argument("--only-leaf", action="store_true", default=False)

args = parser.parse_args()

if args.force_low_count_action or args.optimistic_init:
    args.count_state_action = True

if args.count_state_action or args.count_epsilon:
    args.count = True
# TB
args.tb = not args.no_tb
# Saving the evaluation policies as images
args.eval_images = not args.no_eval_images
# Logging the visitations and exploration bonuses
args.visitations = not args.no_visitations
# No frontier vis
args.frontier = not args.no_frontier
# Model
if args.model == "":
    args.model = args.env

if args.nec:
    args.model = "NEC_" + args.model
if args.distrib:
    args.model += "-Distrib"
if args.model_dqn:
    args.model += "-Model"

print("\n" + "=" * 40)
print(15 * " " + "Settings:" + " " * 15)
print("=" * 40)
for arg in vars(args):
    print(" {}: {}".format(arg, getattr(args, arg)))
print("=" * 40)
print()

NAME_DATE = "{}_{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
LOGDIR = "{}/{}".format(args.logdir, NAME_DATE)

while os.path.exists(LOGDIR):
    LOGDIR += "_"
    NAME_DATE += "_"

args.log_path = LOGDIR

print("Logging to:\n{}\n".format(LOGDIR))

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
if not os.path.exists("{}/logs".format(LOGDIR)):
    os.makedirs("{}/logs".format(LOGDIR))
if not os.path.exists("{}/training".format(LOGDIR)):
    os.makedirs("{}/training".format(LOGDIR))
if not os.path.exists("{}/evals".format(LOGDIR)):
    os.makedirs("{}/evals".format(LOGDIR))
if not os.path.exists("{}/exp_bonus".format(LOGDIR)):
    os.makedirs("{}/exp_bonus".format(LOGDIR))
if not os.path.exists("{}/visitations".format(LOGDIR)):
    os.makedirs("{}/visitations".format(LOGDIR))

with open("{}/settings.txt".format(LOGDIR), "w") as f:
    f.write(str(args))


# Gym Environment
env = gym.make(args.env)
eval_env = gym.make(args.env)

# Atari Wrapping
if args.env.endswith("NoFrameskip-v4"):
    env = wrap_deepmind(env, episode_life=False, clip_rewards=False)
    # TODO: Wrap the evaluation maze differently so it doesn't have frame skips
    eval_env = wrap_deepmind(eval_env, episode_life=False, clip_rewards=False)

# VizDoom Wrapping
if args.env.startswith("ppaquette") or args.env.startswith("Doom"):
    env = wrap_vizdoom(env)
    eval_env = wrap_vizdoom(eval_env)

args.actions = env.action_space.n
args.primitive_actions = args.actions

trainer = Trainer(args, env, eval_env)

start_time = time.time()

trainer.train()

end_time = time.time()

print("Exiting after {}\n".format(time_str(end_time - start_time)))

if args.tar:
    print("Taring directory.")
    with tarfile.open(LOGDIR + ".tar.gz", mode="w:gz") as archive:
        archive.add(LOGDIR, arcname=NAME_DATE)
    print("Finished taring.")
    print("Removing directory {}".format(LOGDIR))
    shutil.rmtree(LOGDIR, ignore_errors=True)
    print("Directory Removed.")
