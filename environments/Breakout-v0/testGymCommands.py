import gym
import time

import helpers
import agent

env = gym.make("Breakout-v0")
env.reset()
a = agent.Agent(env.observation_space.shape, env.action_space.n)

s, r, done = helpers.get_state(env, 0)
print("Shape {}".format(s.shape))

res = a.net(s, debug=True)

print("Shape {}, Value {}".format(res.shape, res))

for t in range(10):
    env.render()
    o, r, d, i = env.step(0)
    time.sleep(0.3)
env.close()


