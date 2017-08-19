import gym
from gym import wrappers

env = gym.make("BipedalWalkerHardcore-v2")
# env = wrappers.Monitor(env, "/tmp/gym-results")
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    if done:
        env.reset()

env.close()

# Gradient test
import mxnet as mx
from mxnet import nd, autograd

ctx = mx.cpu()
x = nd.array([[1, 2], [3, 4]])
x.attach_grad()  # Tell mxnet that we wish to store gradient information in the x NDArray
with autograd.record():  # Tell mxnet to build gradient calculation graph for x. If we don't do this that part of the graph will not build
    y = x * 2
    z = y * x

z.backward()  # Do the backpropagation

print(x.grad)



# Test
import agent
import gym
import mxnet as mx
from mxnet import nd, autograd as ag
import numpy as np

env = gym.make("BipedalWalkerHardcore-v2")
observation = env.reset()
p = agent.Policy()
p.net(nd.array(observation))
