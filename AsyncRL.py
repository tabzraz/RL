import numpy as np
import gym
import tensorflow as tf
import tflearn
import threading

# Parameters
# TODO: Use tf.flags to make cmd line configurable
THREADS = 4
ENV_NAME = "CartPole-v0"
T = 0
T_MAX = 1e6


def actor(env, id, t_max):
    global T, T_MAX

    while T < T_MAX:
        t = 1
        gradients = ()
        s_t = env.reset()
        episode_finished = False
        while (not episode_finished) and t < t_max:
            # env.render()
            a_t = env.action_space.sample()  # TODO: Add network
            s_t, r_t, episode_finished, _ = env.step(a_t)
            t += 1
            # TODO: Do this with a lock for safe concurrent writes
            T += 1
        # TODO: Rest of stuff here


def model():
    inputs = tf.placeholder(tf.float32, shape=[None, 4], name="Observation Input")
    net = tflearn.fully_connected(inputs, 128, activation="relu", weights_init="variance_scaling", name="FC")
    linear_output = tflearn.fully_connected(net, 2, weights_init="variance_scaling", name="Logits")
    softmax_output = tflearn.activations.softmax(linear_output)
    return inputs, softmax_output


# Start the actors
envs = [gym.make(ENV_NAME) for _ in range(THREADS)]
threads = [threading.Thread(target=actor, args=(envs[i], i, 500), daemon=True) for i in range(THREADS)]
for t in threads:
    t.start()
for t in threads:
    t.join()
