from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd as ag

class Agent:
    """An Agent operating in an environment."""

    def __init__(self, policy, env):
        self.__policy = policy
        self.__env = env
        self.__rewards = None
        self.__state = env.reset()

    def state(self):
        print("I'm in state: ", self.__state)

    def act(self):
        self.__policy.decide(self.__state)

    def rollout(self):
        pass


class Policy:
    """A representation of a Policy"""

    def __init__(self):
        import mxnet as mx
        from mxnet import nd, autograd as ag
        ctx = mx.cpu()

        num_inputs = 24
        num_outputs = 4
        batch_size = 64
        num_hidden = 10
        weight_scale = .01

        # Set up the parameters of our 3 layer network
        self.W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale, ctx=ctx)
        self.b1 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=ctx)
        self.W2 = nd.random_normal(shape=(num_hidden, num_hidden), scale=weight_scale, ctx=ctx)
        self.b2 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=ctx)
        self.W3 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale, ctx=ctx)
        self.b3 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=ctx)

        # Gather them and attach gradient calculations
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        for param in self.params:
            param.attach_grad()

    def net(self, X):
        h1_linear = nd.dot(X, self.W1) + self.b1
        h1 = self.relu(h1_linear)
        h2_linear = nd.dot(h1, self.W2) + self.b2
        h2 = self.relu(h2_linear)
        yhat_linear = nd.dot(h2, self.W3) + self.b3
        return yhat_linear

    def relu(self, X):
        return nd.maximum(X, nd.zeros_like(X))

    def sigmoid(self, X):
        return 1 / (1 + nd.exp(-0.5 * X))

    def decide(self, s):
        return net(s)
