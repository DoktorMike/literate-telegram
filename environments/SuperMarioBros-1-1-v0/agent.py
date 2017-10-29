# Imports
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
import gym
from skimage.color import rgb2gray
import cv2

ctx = mx.cpu()

# Create Agent
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0.5
        #  Set the scale for weight initialization and choose the number of hidden units in the fully-connected layer
        self.weight_scale = 0.01
        self.num_fc = 128
        self.num_outputs = action_size
        # Define the weights for the network
        self.W1 = nd.random_normal(shape=(20, 3, 3, 3), scale=self.weight_scale, ctx=ctx)
        self.b1 = nd.random_normal(shape=20, scale=self.weight_scale, ctx=ctx)
        self.W2 = nd.random_normal(shape=(50, 20, 5, 5), scale=self.weight_scale, ctx=ctx)
        self.b2 = nd.random_normal(shape=50, scale=self.weight_scale, ctx=ctx)
        self.W3 = nd.random_normal(shape=(36250, self.num_fc), scale=self.weight_scale, ctx=ctx)
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
        h1_conv = nd.Convolution(data=X, weight=self.W1, bias=self.b1, kernel=(3, 3), num_filter=20)
        if debug: print("h1 shape: %s" % (np.array(h1_conv.shape)))
        h1_activation = self.relu(h1_conv)
        if debug: print("h1 shape: %s" % (np.array(h1_activation.shape)))
        h1 = nd.Pooling(data=h1_activation, pool_type="avg", kernel=(2, 2), stride=(2, 2))
        if debug: print("h1 shape: %s" % (np.array(h1.shape)))

        ########################
        #  Define the computation of the second convolutional layer
        ########################
        h2_conv = nd.Convolution(data=h1, weight=self.W2, bias=self.b2, kernel=(5, 5), num_filter=50)
        if debug: print("h2 shape: %s" % (np.array(h2_conv.shape)))
        h2_activation = self.relu(h2_conv)
        if debug: print("h2 shape: %s" % (np.array(h2_activation.shape)))
        h2 = nd.Pooling(data=h2_activation, pool_type="avg", kernel=(2, 2), stride=(2, 2))
        if debug: print("h2 shape: %s" % (np.array(h2.shape)))

        ########################
        #  Flattening h2 so that we can feed it into a fully-connected layer
        ########################
        h2 = nd.flatten(h2)
        if debug: print("Flat h2 shape: %s" % (np.array(h2.shape)))

        ########################
        #  Define the computation of the third (fully-connected) layer
        ########################
        h3_linear = nd.dot(h2, self.W3) + self.b3
        h3 = self.relu(h3_linear)
        if debug: print("h3 shape: %s" % (np.array(h3.shape)))

        ########################
        #  Define the computation of the output layer
        ########################
        yhat_linear = nd.dot(h3, self.W4) + self.b4
        #yhat = self.softmax(yhat_linear)
        if debug: print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))

        return yhat_linear

    def action(self, state):
        if(np.random.rand() < self.epsilon): return np.random.randint(0, 2, self.action_size)
        q = self.net(state, debug=False)
        q = q[0].asnumpy()
        max_ind = np.argmax(q)
        action = np.eye(self.action_size)[max_ind]
        #print(action)
        #print(max_ind)
        #print(q[max_ind])
        return action, max_ind, q[max_ind]

    def q_value(self, state):
        q = self.net(state, debug=False)
        max_ind = np.argmax(q[0].asnumpy())
        return q[max_ind]


