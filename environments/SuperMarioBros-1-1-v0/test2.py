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

mx.random.seed(2)

def prep_state(state):
    #return cv2.resize(rgb2gray(observation), 128, 112), interpolation=cv2.INTER_LINEAR)
    #plt.imshow(cv2.resize(rgb2gray(observation), (128, 112), interpolation=cv2.INTER_LINEAR))
    #return cv2.resize(rgb2gray(observation), (128, 112), interpolation=cv2.INTER_LINEAR)
    #return state
    #state = state[:, :, (2, 1, 0)]
    return rgb2gray(cv2.resize(state, (128, 112), interpolation=cv2.INTER_LINEAR))

def preprocess(image):
    image = cv2.resize(image, (128, 112), interpolation=cv2.INTER_LINEAR)
    # swap BGR to RGB
    image = image[:, :, (2, 1, 0)]
    # convert to float before subtracting mean
    image = image.astype(np.float32)
    # subtract mean
    image -= np.array([123, 117, 104])
    image /= 255
    # organize as [batch-channel-height-width]
    image = np.transpose(image, (2, 0, 1))
    image = image[np.newaxis, :]
    # convert to ndarray
    image = nd.array(image)
    return image

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def tderror(rt, qval2, qval1, l):
    return nd.nansum((rt + l*qval2 - qval1)**2)



# Setup
num_episodes = 20

# Create main loop
env = gym.make('SuperMarioBros-1-1-v0')
observation = env.reset()
observation = preprocess(observation)
a = agent.Agent(observation.shape, env.action_space.shape)
for episode in range(num_episodes):
    env.render()
    for epoch in range(100):
        #action = env.action_space.sample() # your agent here (this takes random actions)
        with autograd.record():
            qval1 = a.net(observation)
            max_ind = np.argmax(qval1.asnumpy())
            action = np.zeros(a.action_size, dtype=np.int)
            action[max_ind]=1
            observation, reward, done, info = env.step(action)
            observation = preprocess(observation)
            # action2, max_ind2, qval2 = a.action(observation)
            tmpvar = a.action(observation)
            action2 = tmpvar[0]
            max_ind2 = tmpvar[1]
            qval2 = tmpvar[2]
            tdloss = tderror(reward, qval2, mx.nd.take(qval1, mx.nd.array([max_ind])), 0.99)
        tdloss.backward()
        SGD(a.params, 0.01)

        print('Took action {}, Received reward: {}\n'.format(action, reward))
        #print('Gradients: {}'.format(a.b3))
        if done:
            observation = env.reset()
            observation = preprocess(observation)
            print('Epoch {}: Resetting environment\n'.format(epoch))
            break

env.close()

