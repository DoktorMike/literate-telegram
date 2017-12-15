# Random action benchmark for Breakout

# Imports
import gym
from gym import wrappers

import numpy as np
import mxnet as mx
from mxnet import nd, autograd

import cv2

import agent as agent
import helpers

# Valid actions are 0: NoOp, 1: Fire, 2: Left, 3: Right
ACTIONS = [0,1,2,3]

env = gym.make('Breakout-v0')
env.reset()
#env = wrappers.Monitor(env, 'Breakout-v0', force=True)
a = agent.Agent(env.observation_space.shape, env.action_space.n, 0.3)
for i_episode in range(100):
    observation = env.reset()
    observation, reward, done = helpers.get_state(env, 0)
    episode_reward=0
    while (done!=True):
        #env.render()
        #action = env.action_space.sample()
        with autograd.record():
            # Select action
            action1, max_ind1, qval1 = a.action_nd(observation)

            # Do action and observe new state and reward
            observation, reward, done = helpers.get_state(env, max_ind1)
            episode_reward += reward
            if done:
                print("Episode finished after {} episodes".format(i_episode+1))
                #print('Gradient b1: {}'.format(a.b1))
                print('Episode reward: {}'.format(episode_reward))
                #break
                qval2 = 0
            else:
                # Test the best next action
                action2, max_ind2, qval2 = a.action(observation)
            tdloss = helpers.tderror(reward, qval2, qval1, 0.95)
            #print('Chose action {} i.e. {} with reward {} and loss {}'.format(action1,
            #    max_ind1, reward, tdloss.asnumpy()[0]))
            #print('Qval1: {} Qval2: {} TD: {}'.format(qval1.asnumpy(), qval2, reward+0.95*qval2-qval1.asnumpy()))
        tdloss.backward()
        helpers.SGD(a.params, 0.01)

env.close()
#print('Observation: {}'.format(observation.shape))

