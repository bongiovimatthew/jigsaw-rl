# Interface
class Environment(object):

    def reset(self):
        raise NotImplementedError('Inheriting classes must override reset.')

    def actions(self):
        raise NotImplementedError('Inheriting classes must override actions.')

    def step(self):
        raise NotImplementedError('Inheriting classes must override step')


class ActionSpace(object):

    def __init__(self, actions):
        self.actions = actions
        self.n = len(actions)
