# Imports
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from skimage.color import rgb2gray
import numpy as np
import gym
import cv2

ctx = mx.cpu()

# Create Agent
class Agent:
    def __init__(self, state_size, action_size, epsilon):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        # Set the scale for weight initialization and choose the number of hidden units
        # in the fully-connected layer
        self.weight_scale = 0.01
        self.num_fc = 128
        self.num_outputs = action_size
        # Define the weights for the network
        # 20 Filters using a 3x3 kernel for 3 color channels
        self.W1 = nd.random_normal(shape=(20, 3, 2, 3, 3), scale=self.weight_scale, ctx=ctx)
        self.b1 = nd.random_normal(shape=20, scale=self.weight_scale, ctx=ctx)
        # 50 Filters using a 5x5 kernel which will take inputs from 20 Filters from before
        self.W2 = nd.random_normal(shape=(50, 20, 1, 5, 5), scale=self.weight_scale, ctx=ctx)
        self.b2 = nd.random_normal(shape=50, scale=self.weight_scale, ctx=ctx)
        # 50 Filters using a 5x5 kernel which will take inputs from 20 Filters from before
        self.W3 = nd.random_normal(shape=(16200, self.num_fc), scale=self.weight_scale, ctx=ctx)
        self.b3 = nd.random_normal(shape=128, scale=self.weight_scale, ctx=ctx)
        self.W4 = nd.random_normal(shape=(self.num_fc, self.num_outputs), scale=self.weight_scale, ctx=ctx)
        self.b4 = nd.random_normal(shape=self.num_outputs, scale=self.weight_scale, ctx=ctx)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]
        for param in self.params:
            param.attach_grad()

    def relu(self, X):
        return nd.maximum(X, nd.zeros_like(X))

    def softmax(self, y):
        exp = nd.exp(y - nd.max(y))
        partition = nd.sum(exp, axis=0, exclude=True).reshape((-1, 1))
        return exp / partition

    def net(self, X, debug=False):
        ########################
        #  Define the computation of the first convolutional layer
        ########################
        h1_conv = nd.Convolution(data=X, weight=self.W1, bias=self.b1, kernel=(2, 3, 3), num_filter=20)
        h1_activation = self.relu(h1_conv)
        h1 = nd.Pooling(data=h1_activation, pool_type="avg", kernel=(2, 2, 2), stride=(2, 2, 2))

        ########################
        #  Define the computation of the second convolutional layer
        ########################
        h2_conv = nd.Convolution(data=h1, weight=self.W2, bias=self.b2, kernel=(1, 5, 5), num_filter=50)
        h2_activation = self.relu(h2_conv)
        h2 = nd.Pooling(data=h2_activation, pool_type="avg", kernel=(1, 2, 2), stride=(1, 2, 2))

        ########################
        #  Flattening h2 so that we can feed it into a fully-connected layer
        ########################
        h2 = nd.flatten(h2)

        ########################
        #  Define the computation of the third (fully-connected) layer
        ########################
        h3_linear = nd.dot(h2, self.W3) + self.b3
        h3 = self.relu(h3_linear)

        ########################
        #  Define the computation of the output layer
        ########################
        yhat_linear = nd.dot(h3, self.W4) + self.b4
        #yhat = self.softmax(yhat_linear)

        return yhat_linear

    def action(self, state):
        q = self.net(state, debug=False)
        q = q[0].asnumpy()
        max_ind = np.argmax(q)
        action = np.zeros(self.action_size, dtype=np.int)
        if (np.random.rand() < self.epsilon):
            #action = nd.one_hot(nd.array(np.random.randint(0,self.action_size,1)), self.action_size)
            max_ind = np.random.randint(0, self.action_size, 1)[0]
        action[max_ind] = 1
        #print(action)
        #print(max_ind)
        #print(q[max_ind])
        return action, max_ind, q[max_ind]

    def action_nd(self, state):
        q = self.net(state, debug=False)
        max_ind = np.int(np.argmax(q[0].asnumpy()))
        action = np.zeros(self.action_size, dtype=np.int)
        if (np.random.rand() < self.epsilon):
            #action = nd.one_hot(nd.array(np.random.randint(0,self.action_size,1)), self.action_size)
            max_ind = np.int(np.random.randint(0, self.action_size, 1)[0])
        action[max_ind] = 1
        #print(action)
        #print(max_ind)
        #print(q)
        return action, max_ind, q[0][max_ind]

    def q_value(self, state):
        q = self.net(state, debug=False)
        max_ind = np.argmax(q[0].asnumpy())
        return q[max_ind]


