from IAgent import Agent 
class QLearningAgent(Agent):
    
    def __init__(self, actions, epsilon=0.01, alpha=0.5, gamma=1):
        super(QLearningAgent, self).__init__(actions)
        
        ## TODO 1
        ## Initialize empty dictionary here
        ## In addition, initialize the value of epsilon, alpha and gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.QDict = {}
        
    def stateToString(self, state):
        mystring = ""
        if np.isscalar(state):
            mystring = str(state)
        else:
            for digit in state:
                mystring += str(digit)
        return mystring    
    
    def act(self, state):
        stateStr = self.stateToString(state)      
        action = np.random.randint(0, self.num_actions) 
        
        ## TODO 2
        ## Implement epsilon greedy policy here
        choice = None
        if self.epsilon == 0:
            choice = 0
        elif self.epsilon == 1:
            choice = 1
        else:
            choice = np.random.binomial(1, self.epsilon)
            
        if choice == 1:
            return action
        else:
            applicableQValues = [QDictKey for QDictKey in self.QDict.keys() if QDictKey.state==stateStr and self.QDict[QDictKey] != 0]
            if len(applicableQValues) == 0:
                return action
            
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
        
        ## TODO 3
        ## Implement the q-learning update here
        Q1 = Q_Tuple(state1Str, action1)
        
        initQ1Value = 0
        maxQStateValue = 0

        if Q1 in self.QDict.keys():
            initQ1Value = self.QDict[Q1]
                    
        applicableQValues = [QDictKey for QDictKey in self.QDict.keys() if QDictKey.state==state2Str and self.QDict[QDictKey] != 0 ]
        if len(applicableQValues) > 0:
#             print("len: " + str(len(applicableQValues)))
            maxQStateValue = self.QDict[applicableQValues[0]]
            for QDictKey in applicableQValues[1:]:
                if self.QDict[QDictKey] > maxQStateValue:
                    maxQStateValue = self.QDict[QDictKey]
                    
        td_target = reward + self.gamma * maxQStateValue
        td_delta = td_target - initQ1Value
        self.QDict[Q1] = initQ1Value + self.alpha * td_delta 