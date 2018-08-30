from ActionChoosers.IActionChooser import IActionChooser
import numpy as np
import random as r

class EpsilonGreedyActionChooser(IActionChooser):

    def __init__(self, brain):
        self.Brain = brain
        self.debugMode = True

    def get_brain(self):
        return self.Brain

    def action(self, state, epsilon):
        act = 0

        n = self.Brain.get_num_actions()


        if state is None:
            act = r.randint(0, n - 1)
        else:
            # Decide to explore or not.
            explore = False

            if epsilon != -1:
                explore = np.random.binomial(1, epsilon)

            if explore:
                act = r.randint(0, n-1)
                print("Exploring : act: {0}".format(act))
            else:
                act = self.exploit(state)
                print("Exploiting : act: {0}".format(act))

        return act

    def exploit(self, state):
        n = self.Brain.get_num_actions()
        prob_vec = self.Brain.action_probabilities(state)

        if self.debugMode:
            print("Probability vector: ", prob_vec)

        maxProbability = prob_vec[0]
        possibleActions = [0]
        for i in range(1, n):
            if prob_vec[i] > maxProbability:
                maxProbability = prob_vec[i]
                possibleActions = [i]
            elif prob_vec[i] == maxProbability:
                possibleActions.append(i)

        return possibleActions[np.random.randint(0, len(possibleActions))]
