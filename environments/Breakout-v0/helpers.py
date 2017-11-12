
# Imports
import gym
from gym import wrappers

import numpy as np
import mxnet as mx
from mxnet import nd, autograd

import cv2
from skimage.color import rgb2gray

# Helper functions
def preprocess(obs):
    # resize
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_LINEAR)
    # gray scale
    obs = rgb2gray(obs)
    obs = obs[np.newaxis, :]
    # rescale
    obs = obs.astype(np.float32)
    obs /= 255
    # organize as [batch-channel-depth-height-width]
    #obs = np.transpose(obs, (2, 0, 1))
    obs = obs[np.newaxis, :]
    obs = obs[np.newaxis, :]
    obs = np.transpose(obs, (0, 2, 1, 3, 4))
    # return nd.array(obs)
    return obs

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def tderror(rt, qval2, qval1, l):
    return nd.nansum((rt + l*qval2 - qval1)**2)

def get_state(env, action):
    observation, reward, done, info = env.step(action)
    observation = preprocess(observation)
    o = observation
    r = reward
    for i in range(3):
        observation, reward, done, info = env.step(action)
        observation = preprocess(observation)
        #o = np.vstack((o, observation))
        o = np.concatenate((o, observation), axis=2)
        r += reward
    return nd.array(o), r, done


