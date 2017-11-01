# Random action benchmark for Breakout

# Imports
import gym
from gym import wrappers

import numpy as np
import mxnet as mx
from mxnet import nd, autograd

import cv2

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

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def tderror(rt, qval2, qval1, l):
    return nd.nansum((rt + l*qval2 - qval1)**2)

env = gym.make('Breakout-v0')
env.reset()
#env = wrappers.Monitor(env, 'Breakout-v0', force=True)
a = agent.Agent(env.observation_space.shape, env.action_space.n)
for i_episode in range(10):
    observation = env.reset()
    observation = preprocess(observation)
    while (True):
        #env.render()
        #action = env.action_space.sample()
        with autograd.record():
            # Select action
            action1, max_ind1, qval1 = a.action_nd(observation)

            # Do action and observe new state and reward
            observation, reward, done, info = env.step(max_ind1)
            if done:
                print("Episode finished after {} episodes".format(i_episode+1))
                print('Gradient b1: {}'.format(a.b1))
                break
            observation = preprocess(observation)

            # Test the best next action
            action2, max_ind2, qval2 = a.action(observation)
            tdloss = tderror(reward, qval2, qval1, 0.95)
            print('Chose action {} i.e. {} with reward {} and loss {}'.format(action1,
                max_ind1, reward, tdloss.asnumpy()[0]))
            print('Qval1: {} Qval2: {} TD: {}'.format(qval1.asnumpy(), qval2, reward+0.95*qval2-qval1.asnumpy()))
        tdloss.backward()
        SGD(a.params, 1)

env.close()
#print('Observation: {}'.format(observation.shape))

