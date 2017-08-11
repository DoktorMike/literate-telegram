import gym
from gym import wrappers

env = gym.make("BipedalWalkerHardcore-v2")
#env = wrappers.Monitor(env, "/tmp/gym-results")
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    if done:
        env.reset()

env.close()

class Agent:
    """An Agent operating in an environment."""
    def __init__(self, policy, env):
        self.__policy = policy
        self.__env = env
        self.__rewards = None
        self.__state = env.reset()
    def state(self):
        print("I'm in state: ", self.__state)


env = gym.make("BipedalWalkerHardcore-v2")
a = Agent(1, env)
a.state()
