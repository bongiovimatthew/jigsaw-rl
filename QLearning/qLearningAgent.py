from IAgent import Agent 
from collections import namedtuple
from metricsManager import MetricsManager

import numpy as np
Q_Tuple = namedtuple('Q_Tuple', 'state action')

class QLearningAgent(Agent):
    
    def __init__(self, actions, epsilon=0.6, alpha=0.5, gamma=1):
        super(QLearningAgent, self).__init__(actions)
        
        # epsilon is the rate of exploration - higher epsilon means more exploring 
        self.epsilon = epsilon

        # the rate at which we learn
        self.alpha = alpha

        # the rate at which you discount a reward that is further away in steps - higher gamma means 
        #  a future reward is valued more highly at the current time 
        self.gamma = gamma

        # the dictionary of Q values for state,action pairs 
        # Key is a tuple of (state, action), value is the Q value of that pair 
        self.QDict = {}

        self.MetricsManager = MetricsManager()
        
    def stateToString(self, state):
        mystring = ""
        if np.isscalar(state):
            mystring = str(state)
        else:
            for digit in state:
                mystring += str(digit)
        return mystring    
    
    def act(self, state):

        self.MetricsManager.displayMetrics()

        stateStr = self.stateToString(state)      
        action = np.random.randint(0, self.num_actions) 
        
        ## Implement epsilon greedy policy here
        choice = None
        if self.epsilon == 0:
            choice = 0
        elif self.epsilon == 1:
            choice = 1
        else:
            choice = np.random.binomial(1, self.epsilon)
            
        if choice == 1:
            self.MetricsManager.metric_totalExplores += 1
            return action
        else:
            self.MetricsManager.metric_totalQarray += 1 
            applicableQValues = [QDictKey for QDictKey in self.QDict.keys() if QDictKey.state==stateStr and self.QDict[QDictKey] != 0]
            if len(applicableQValues) == 0:
                self.MetricsManager.metric_qarrayMiss += 1
                return action
            
            self.MetricsManager.metric_qarrayHit += 1
            self.MetricsManager.metric_totalQarrayLengthOnHit += len(applicableQValues)

            maxQValue = self.QDict[applicableQValues[0]]
            relevantQStates = [applicableQValues[0]]
            for QDictKey in applicableQValues[1:]:
                if self.QDict[QDictKey] > maxQValue:
                    maxQValue = self.QDict[QDictKey]
                    relevantQStates = [QDictKey]
                elif self.QDict[QDictKey] == maxQValue:
                    relevantQStates.append(QDictKey)                
            
            QToPick = np.random.randint(0, len(relevantQStates))
            return relevantQStates[QToPick].action
        
        return action
    
    def learn(self, state1, action1, reward, state2, done):
        state1Str = self.stateToString(state1)
        state2Str = self.stateToString(state2)

        self.MetricsManager.metric_totalReward += reward
        self.MetricsManager.metric_rewardOps += 1
        
        ## Implement the q-learning update here
        Q1 = Q_Tuple(state1Str, action1)
        
        initQ1Value = 0
        maxQStateValue = 0

        if Q1 in self.QDict.keys():
            initQ1Value = self.QDict[Q1]
                    
        applicableQValues = [QDictKey for QDictKey in self.QDict.keys() if QDictKey.state==state2Str and self.QDict[QDictKey] != 0 ]
        if len(applicableQValues) > 0:
            maxQStateValue = self.QDict[applicableQValues[0]]
            for QDictKey in applicableQValues[1:]:
                if self.QDict[QDictKey] > maxQStateValue:
                    maxQStateValue = self.QDict[QDictKey]
                    
        td_target = reward + self.gamma * maxQStateValue
        td_delta = td_target - initQ1Value
        self.QDict[Q1] = initQ1Value + self.alpha * td_delta 