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

def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)

def net(X):
    h1_linear = nd.dot(X, W1) + b1
    h1 = relu(h1_linear)
    h2_linear = nd.dot(h1, W2) + b2
    h2 = relu(h2_linear)
    yhat_linear = nd.dot(h2, W3) + b3
    return yhat_linear

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    data_iterator.reset()
    for i, batch in enumerate(data_iterator):
        data = batch.data[0].as_in_context(ctx).reshape((-1, 784))
        label = batch.label[0].as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

epochs = 10
moving_loss = 0.
learning_rate = .001

for e in range(epochs):
    train_data.reset()
    for i, batch in enumerate(train_data):
        data = batch.data[0].as_in_context(ctx).reshape((-1, 784))
        label = batch.label[0].as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)

        ##########################
        #  Keep a moving average of the losses
        ##########################
        if (i == 0) and (e == 0):
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, moving_loss, train_accuracy, test_accuracy))
