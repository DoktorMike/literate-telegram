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
        __policy.decide(__state)
    def rollout(self):
        pass

class Policy:
    """A representation of a Policy"""
    def __init__(self):
        self.__params
    def decide(s):
        pass


