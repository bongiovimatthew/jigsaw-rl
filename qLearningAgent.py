from IAgent import Agent 
from collections import namedtuple
import matplotlib.pyplot as plt

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

        self.metric_qarrayHit = 0.0
        self.metric_qarrayMiss = 0.0
        self.metric_totalQarray = 0.0
        self.metric_totalQarrayLengthOnHit = 0.0

        # totalQarrayLengthOnHit / qarrayHit gives average qarray length on hit

        self.metric_totalReward = 0.0
        self.metric_rewardOps = 0.0

        self.metric_totalExplores = 0.0
        
        #self.fig = plt.figure()
        #self.fig.show()

        
    def stateToString(self, state):
        mystring = ""
        if np.isscalar(state):
            mystring = str(state)
        else:
            for digit in state:
                mystring += str(digit)
        return mystring    
    
    def displayMetrics(self):

        
        #self.fig.text(2, 6, 'testing', fontsize=15)
        

        hitToMissRatio = 0.0
        if self.metric_qarrayMiss != 0: 
            hitToMissRatio = self.metric_qarrayHit / self.metric_qarrayMiss

        print("-----------------------------------------------------------------------")
        print("Num QArray Hits: {0}".format(self.metric_qarrayHit))
        print("Num QArray Miss: {0}".format(self.metric_qarrayMiss))
        print("Hit to Miss Ratio: {0}".format(hitToMissRatio))

        print("Num Explores: {0}".format(self.metric_totalExplores))
        print("Num Total QArray Ops (num exploits): {0}".format(self.metric_totalQarray))

        exploitExploreRatio = 0.0
        if self.metric_totalExplores != 0: 
            exploitExploreRatio = self.metric_totalQarray / self.metric_totalExplores

        print("Exploit / Explore Ratio: {0}".format(exploitExploreRatio))

        avgQArrayLen = 0
        avgReward = 0
        if self.metric_qarrayHit != 0: 
            avgQArrayLen = self.metric_totalQarrayLengthOnHit / self.metric_qarrayHit

        if self.metric_rewardOps != 0: 
            avgReward = self.metric_totalReward / self.metric_rewardOps

        print("Avg Num Qs in QArray on Hits: {0}".format(avgQArrayLen))
        print("Avg Reward: {0}".format(avgReward))
        print("-----------------------------------------------------------------------")
        print()
        print()
        return 

    def act(self, state):

        self.displayMetrics()

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
            self.metric_totalExplores += 1
            return action
        else:
            self.metric_totalQarray += 1 
            applicableQValues = [QDictKey for QDictKey in self.QDict.keys() if QDictKey.state==stateStr and self.QDict[QDictKey] != 0]
            if len(applicableQValues) == 0:
                self.metric_qarrayMiss += 1
                return action
            
            self.metric_qarrayHit += 1
            self.metric_totalQarrayLengthOnHit += len(applicableQValues)

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

        self.metric_totalReward += reward
        self.metric_rewardOps += 1
        
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