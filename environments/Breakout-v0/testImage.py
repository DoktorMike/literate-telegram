
# Imports
import gym
from gym import wrappers

import numpy as np
import mxnet as mx
from mxnet import nd, autograd

import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray

import agent

# Helper functions
def preprocess(obs):
    # resize
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_LINEAR)
    # rescale
    obs = obs.astype(np.float32)
    obs /= 255
    # organize as [batch-channel-height-width]
    obs = np.transpose(obs, (2, 0, 1))
    obs = obs[np.newaxis, :]
    return nd.array(obs)

# Helper functions
def preprocess2(obs):
    # resize
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_LINEAR)
    # rescale
    obs = obs.astype(np.float32)
    obs /= 255
    return obs

env = gym.make("Breakout-v0")

o = env.reset()
o, r, d, i = env.step(env.action_space.sample())
o, r, d, i = env.step(env.action_space.sample())
o, r, d, i = env.step(env.action_space.sample())

#plt.imshow(cv2.resize(rgb2gray(o), (105, 80), interpolation=cv2.INTER_LINEAR))
#plt.imshow(cv2.resize(rgb2gray(o), (84, 84), interpolation=cv2.INTER_LINEAR))
#plt.imshow(cv2.resize(o, (84, 84), interpolation=cv2.INTER_LINEAR))
plt.imshow(preprocess2(o))
plt.show()

