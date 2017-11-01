import gym
import gym_pull
gym_pull.pull('github.com/ppaquette/gym-super-mario')        # Only required once, envs will be loaded with import gym_pull afterwards
env = gym.make('ppaquette/SuperMarioBros-1-1-v0')


# Imports
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
import gym
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2
import agent

mx.random.seed(1)

env = gym.make('SuperMarioBros-1-1-v0')
observation = env.reset()
for i in [1,2,3,4,5,6]:
    observation, reward, done, info = env.step(env.action_space.sample())

a = env.reset()
