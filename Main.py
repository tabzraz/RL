import argparse
import datetime
import gym
import os
import time
import Envs
from Utils.Utils import time_str
from RL_Trainer_PyTorch import Trainer

# Argument passing stuff is here
# TODO: Spread these out into useful groups and provide comments
parser = argparse.ArgumentParser(description="RL Agent Trainer")
parser.add_argument("--t-max", type=int, default=int(1e4))
parser.add_argument("--env", type=str, default="Maze-2-v0")
parser.add_argument("--logdir", type=str, default="Logs")
parser.add_argument("--name", type=str, default="nn")
parser.add_argument("--exp-replay-size", "--xp", type=int, default=int(1e4))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--model", type=str, default="")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--render", action="store_true", default=False)
parser.add_argument("--slow-render", action="store_true", default=False)
parser.add_argument("--epsilon-steps", "--eps-steps", type=int, default=int(7e3))
parser.add_argument("--epsilon-start", "--eps-start", type=float, default=1.0)
parser.add_argument("--epsilon-finish", "--eps-finish", type=float, default=0.1)
parser.add_argument("--batch-size", "--batch", type=int, default=32)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--target", type=int, default=100)
parser.add_argument("--count", action="store_true", default=False)
parser.add_argument("--beta", type=float, default=0.01)
parser.add_argument("--exploration-steps", "--exp-steps", type=int, default=int(5e4))
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
parser.add_argument("--cts-size", type=int, default=7)
parser.add_argument("--cts-conv", action="store_true", default=False)
parser.add_argument("--exp-bonus-save", type=float, default=0.75)
parser.add_argument("--clip-reward", action="store_true", default=False)
parser.add_argument("--options", type=str, default="Primitive")
parser.add_argument("--num-macros", type=int, default=10)
parser.add_argument("--max-macro-length", type=int, default=10)
parser.add_argument("--macro-seed", type=int, default=12)
parser.add_argument("--train-primitives", action="store_true", default=False)
args = parser.parse_args()

# TB
args.tb = not args.no_tb
# Saving the evaluation policies as images
args.eval_images = not args.no_eval_images
# Model
if args.model == "":
    args.model = args.env

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

args.logs_path = LOGDIR

print("Logging to:\n{}\n".format(LOGDIR))

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
if not os.path.exists("{}/logs".format(LOGDIR)):
    os.makedirs("{}/logs".format(LOGDIR))
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
args.actions = env.action_space.n
args.primitive_actions = args.actions

trainer = Trainer(args, env)

start_time = time.time()

trainer.train()

end_time = time.time()

print("\nExiting after {}\n".format(time_str(end_time - start_time)))
