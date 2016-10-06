import numpy as np
from tqdm import tqdm
import gym
from Replay import ExpReplay
from keras.layers import Input, Dense, Activation, Dropout, \
    Convolution2D, Flatten
from keras.models import Model
from keras.optimizers import Adamax
import os
import json
import datetime
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque


# Parameters
ENV_NAME = "SpaceInvaders-frame-v0"
ALGORITHM = "DQN"
MODEL_DESC = "Resize, grayscale and framestack preprocessing, 3 Layer ConvNet, " + \
    "Dropout, 1 Layer FC ReLU"
EXP_SIZE = 100000
EXP_STEPS = 100000
GAMMA = 0.99
EPISODES = 250
MAX_STEPS = 9000
UPDATES_PER_STEP = 1
TARGET_NETWORK_UPDATE = 10000
BATCH_SIZE = 31
ONLINE_TRAIN = True
MODEL_CHECKPOINT = 25
DIRECTORY_INFO = "Testing"
ACTION_REPEAT = 4
######


def atari_image_model():

    inputs = Input(shape=(1 * ACTION_REPEAT, 105, 80))  # RGB Image 210 x 160

    # Max pool instead of pre-processing
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    x = Convolution2D(32, 5, 5, subsample=(3, 3))(inputs)
    x = Activation("relu")(x)
    x = Convolution2D(64, 3, 3, subsample=(2, 2))(x)
    x = Activation("relu")(x)
    x = Convolution2D(64, 3, 3, subsample=(1, 1))(x)
    x = Activation("relu")(x)
    # x = Convolution2D(96, 3, 3, subsample=(2, 2))(x)
    # x = Activation("relu")(x)
    # x = Convolution2D(128, 3, 3, subsample=(1, 1))(x)
    # x = Activation("relu")(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)

    x = Dense(256, init="glorot_normal")(x)
    x = Activation("relu")(x)
    # x = Dense(128, init="glorot_normal")(x)
    # x = Activation("relu")(x)

    outputs = Dense(env.action_space.n, init="uniform")(x)

    model = Model(input=inputs, output=outputs, name="DQN")
    optimiser = Adamax(lr=10**(-4))
    model.compile(optimizer=optimiser, loss="mse")

    return model

# Setup environment and networks
print("Making environment:", ENV_NAME)
env = gym.make(ENV_NAME)

print("Compiling models")
dqn = atari_image_model()
dqn.summary()
old_dqn = atari_image_model()
old_dqn.set_weights(dqn.get_weights())
replay = ExpReplay.ExperienceReplay(EXP_SIZE)

info = {}
info["model"] = dqn.to_json()
info["model_desc"] = MODEL_DESC
info["environment"] = ENV_NAME
info["replay_size"] = EXP_SIZE
info["replay_exploration_steps"] = EXP_STEPS
info["gamma"] = GAMMA
info["episodes"] = EPISODES
info["max_steps_per_episode"] = MAX_STEPS
info["updates_per_step"] = UPDATES_PER_STEP
info["target_update_steps"] = TARGET_NETWORK_UPDATE
info["minibatch_size"] = BATCH_SIZE
info["online_training"] = ONLINE_TRAIN
info["action_repeat"] = ACTION_REPEAT

# Setup logging directorties
print("Creating log directory")
directory = ENV_NAME + "___" + ALGORITHM + "___" + DIRECTORY_INFO + "___" + \
    datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")

if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

with open("info.json", "w") as fp:
    json.dump(info, fp)


# Frame processing stuff
def process_frame(frame, get_state=False):
    new_frame = resize(rgb2gray(frame), (105, 80))
    # Divide by 255 to put in [0,1] then take away 0.5 to kinda subtract mean
    new_frame /= 255.0
    new_frame -= 0.5
    prev_frames.popleft()
    prev_frames.append(new_frame)
    if get_state:
        return np.stack(prev_frames, axis=0)

# Init to zeroes
prev_frames = deque([np.zeros(shape=(105, 80))
                     for _ in range(ACTION_REPEAT)])

# Start the algorithm
print("Starting episodes")

# Deep Q Learning
total_steps = 0

for i in tqdm(range(1, EPISODES + 1)):
    frame = env.reset()
    state = process_frame(frame, get_state=True)
    epsilon = 0.1 + 0.9 * ((EPISODES - i) / EPISODES)
    total_reward = 0
    steps = 0
    histories = []

    for _ in range(MAX_STEPS):

        # Pick Action
        action = None
        if total_steps < EXP_STEPS or np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            actions = dqn.predict(np.swapaxes(np.swapaxes(
                state, 1, 2), 0, 1).reshape(1, 1 * ACTION_REPEAT, 105, 80))
            action = np.argmax(actions)

        # Make a step
        for a_i in range(ACTION_REPEAT):
            frame_new, reward, done, info = env.step(action)
            total_reward += reward
            if a_i == ACTION_REPEAT - 1:
                state_new = process_frame(frame_new, get_state=True)
            else:
                process_frame(frame_new)

        replay.Add_Exp(state, action, reward, state_new, done)

        batch = replay.Sample(BATCH_SIZE)
        # Train on the current experience and the history
        if ONLINE_TRAIN:
            batch.append((state, action, reward, state_new, done))

        # Setup target vectors
        y = np.zeros(shape=(len(batch), env.action_space.n))
        for index, batch_item in enumerate(batch):
            st, at, rt, snew, term = batch_item
            q = dqn.predict(np.swapaxes(np.swapaxes(st, 1, 2), 0, 1).reshape(
                1, 1 * ACTION_REPEAT, 105, 80))
            q_target = old_dqn.predict(np.swapaxes(np.swapaxes(
                snew, 1, 2), 0, 1).reshape(1, 1 * ACTION_REPEAT, 105, 80))
            y[index] = q
            if term:
                y[index][at] = rt
            else:
                y[index][at] = rt + GAMMA * np.max(q_target)

        for _ in range(UPDATES_PER_STEP):
            hist = dqn.train_on_batch(
                np.array(list(map(lambda tups: np.swapaxes(np.swapaxes(tups[0], 1, 2), 0, 1).reshape(1 * ACTION_REPEAT, 105, 80), batch))), y)
            histories.append(hist)

        state = state_new
        steps += 1
        total_steps += 1

        if total_steps % TARGET_NETWORK_UPDATE == 0:
            old_dqn.set_weights(dqn.get_weights())

        if done:
            break

    # Save the results from this episode to the relevant files
    with open("rewards.info", mode="a") as fp:
        fp.write(str(total_reward) + "\n")
    with open("loss.info", mode="a") as fp:
        for h in histories[:-1]:
            fp.write(str(h) + ", ")
        fp.write(str(histories[-1]) + "\n")
    with open("steps.info", mode="a") as fp:
        fp.write(str(steps) + "\n")
    if i % MODEL_CHECKPOINT == 0:
        dqn.save_weights("weights_" + str(i) + ".h5")


print("Saving final model weights")
dqn.save_weights("weights_" + str(EPISODES) + "_end.h5")
