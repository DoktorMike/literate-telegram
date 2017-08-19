from __future__ import print_function

import gym
from gym import wrappers
import mxnet as mx
from mxnet import nd, autograd as ag
import numpy as np

ctx = mx.cpu()

env = gym.make("BipedalWalkerHardcore-v2")
# env = wrappers.Monitor(env, "/tmp/gym-results")
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    if done:
        env.reset()

env.close()
