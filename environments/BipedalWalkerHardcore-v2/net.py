from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
ctx = mx.cpu()

mnist = mx.test_utils.get_mnist()
num_inputs = 784
num_outputs = 10
batch_size = 64
train_data = mx.io.NDArrayIter(mnist["train_data"], mnist["train_label"],
                               batch_size, shuffle=True)
test_data = mx.io.NDArrayIter(mnist["test_data"], mnist["test_label"],
                              batch_size, shuffle=True)

#######################
#  Set some constants so it's easy to modify the network later
#######################
num_hidden = 256
weight_scale = .01

#######################
#  Allocate parameters for the first hidden layer
#######################
W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale, ctx=ctx)
b1 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=ctx)

#######################
#  Allocate parameters for the second hidden layer
#######################
W2 = nd.random_normal(shape=(num_hidden, num_hidden), scale=weight_scale, ctx=ctx)
b2 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=ctx)

#######################
#  Allocate parameters for the output layer
#######################
W3 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale, ctx=ctx)
b3 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=ctx)

params = [W1, b1, W2, b2, W3, b3]

for param in params:
    param.attach_grad()

def relu(X):
    return nd.maximum(X, nd.zeros_like(X))

def sigmoid(X):
    return 1/(1+nd.exp(-0.5*X))

def cross_entropy(yhat, y):
    return - nd.nansum(y * nd.log(yhat), axis=0, exclude=True)

def net(X):
    #######################
    #  Compute the first hidden layer
    #######################
    h1_linear = nd.dot(X, W1) + b1
    h1 = relu(h1_linear)

    #######################
    #  Compute the second hidden layer
    #######################
    h2_linear = nd.dot(h1, W2) + b2
    h2 = relu(h2_linear)

    #######################
    #  Compute the output layer.
    #  We will omit the softmax function here
    #  because it will be applied
    #  in the softmax_cross_entropy loss
    #######################
    yhat_linear = nd.dot(h2, W3) + b3
    return yhat_linear

