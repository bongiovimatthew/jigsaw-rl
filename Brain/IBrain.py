class IBrain:

    def action_probabilities(self, state):
        pass

    def state_value(self, state):
        pass

    def train(self, states, actions, rewards):
        pass

    def get_num_actions(self):
        pass
