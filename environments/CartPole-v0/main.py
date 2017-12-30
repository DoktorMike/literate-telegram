# Imports
from __future__ import print_function
import gym
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np

# Constants
num_inputs = 4
num_outputs = 2
epsilon = 0.8
learning_rate = 0.001
discount = 0.99
gradient_max = 1

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
def sgd(params, lr, clip=True):
    for param in params:
        if (clip):
            l2norm = mx.nd.norm(param.grad)
            if l2norm > gradient_max:
                param.grad[:] = 0
        param[:] = param - lr * param.grad


def square_loss(yhat, y):
    """
    Calculate the squared loss where yhat is our estimate and y is our target
    :param yhat:
    :param y:
    :return:
    """
    return nd.mean((yhat - y) ** 2)


def net(X):
    return mx.nd.dot(X, w) + b


def action(o):
    if np.random.rand() > epsilon:
        return int(nd.argmax(o, axis=0).asscalar().astype(int))  # Add epsilon greedy step
    else:
        return env.action_space.sample()


def gen_label(q, r, d):
    a = r + d * q
    return a.asscalar().astype(float)


env = gym.make('CartPole-v0')
debug = 0
for i in range(40000):
    R, N = 0, 0
    s1 = env.reset()
    o1 = net(nd.array(s1, ctx=model_ctx))
    a1 = action(o1)
    #q1 = nd.max(o1) # Wrong the q value has to be the one according to the action we chose!
    q1 = o1[a1]
    for j in range(1000):
        # env.render()
        # o, r, done, info = env.step(env.action_space.sample()) # take a random action
        s2, r, done, info = env.step(a1)
        o2 = net(nd.array(s2, ctx=model_ctx))
        a2 = int(nd.argmax(o2, axis=0).asscalar().astype(int))
        q2 = o2[a2]
        # ylab = gen_label(q2, r, discount) if not done else -100
        ylab = gen_label(q2, r, discount) if not done else r
        if (debug>1):
            print("s1: {}, o1: {}, a1: {}, q1: {}".format(s1, o1.asnumpy(), a1, q1.asnumpy()))
            print("s2: {}, o2: {}, a2: {}, q2: {}".format(s2, o2.asnumpy(), a2, q2.asnumpy()))
            print("y: {}, ylab: {}".format(q1.asnumpy(), ylab))
            print("Parameters w: {}".format(w.asnumpy()))
            print("Parameters b: {}".format(b.asnumpy()))
            input("Press Enter to continue...")
        with autograd.record():
            o1 = net(nd.array(s1, ctx=model_ctx))
            a1 = action(o1)
            q1 = o1[a1]
            loss = square_loss(q1, ylab)
        loss.backward()
        sgd(params, learning_rate, False)
        R += r
        N += 1

        s1, a1 = s2, a2
        if done == True:
            if i % 100 == 0:
                print("Episode {} finished after {} steps. Total reward: {}, Epsilon: {}".format(i, N, R, epsilon))
                if (debug>0):
                    print("Parameters w: {}".format(w.asnumpy()))
                    print("Parameters b: {}".format(b.asnumpy()))
            break
    if i % 100 == 0:
        epsilon = epsilon * 0.9 if epsilon > 0.05 else 0.05



