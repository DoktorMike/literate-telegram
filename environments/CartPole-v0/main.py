# Imports
from __future__ import print_function
import gym
import mxnet as mx
from mxnet import nd, autograd, gluon

# Constants
num_inputs = 4
num_outputs = 2
epsilon = 0.2

# Define places to compute
data_ctx = mx.cpu()
model_ctx = mx.cpu()

# Define paramters that are fucking global
w = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
b = nd.random_normal(shape=num_outputs, ctx=model_ctx)
params = [w, b]

# Attach gradients
for param in params:
        param.attach_grad()

# Functions for training
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def square_loss(yhat, y):
    return nd.mean((yhat - y) ** 2)

def net(X):
    return mx.nd.dot(X, w) + b

env = gym.make('CartPole-v0')
for i in range(20):
    R, N = 0, 0
    s1 = env.reset()
    q1 = net(nd.array(s1))
    a1 = nd.argmax(q1, axis=0) # Add epsilon greedy step
    for _ in range(1000):
        env.render()
        #o, r, done, info = env.step(env.action_space.sample()) # take a random action
        s2, r, done, info = env.step(a1.asscalar()) # take a random action
        q2 = net(nd.array(s2))
        a2 = nd.argmax(q2, axis=0) # Add epsilon greedy step
        R += r
        N += 1

        a1 = a2
        if done==True:
            env.reset()
            print("Episode {} finished after {} steps. Total reward: {}".format(i, R, N))
            break

