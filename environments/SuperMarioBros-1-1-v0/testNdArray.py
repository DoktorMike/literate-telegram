import mxnet as mx
from mxnet import nd, autograd
import numpy as np

a = [1,2,3,4]
a1 = mx.nd.array(a)


def hello():
    return [1,2,3], 54, np.nan

q = nd.array([1,2,4,3])
max_ind = mx.nd.array([np.argmax(q.asnumpy())])
mx.nd.take(q, max_ind)
action = nd.zeros(6)
action[max_ind.astype(int)[0]] = 1
action[3] = 4
action[np.int(np.argmax(action.asnumpy()))] = 3

nd.zeros(6, dtype=np.int)

