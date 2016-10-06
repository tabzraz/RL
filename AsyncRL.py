import numpy as np
import gym
import tensorflow as tf

ENV_NAME = "FlappyBird-v0"

env = gym.make(ENV_NAME)
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
