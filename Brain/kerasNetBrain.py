from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras.models import load_model

from Brain.IBrain import IBrain 

class KerasNetBrain(IBrain): 

    def __init__(self, num_actions, lr, stateShape):

        self.num_actions = num_actions
        self.lr = lr
        self.debugMode = True
        self.STATE_WIDTH = stateShape[0]
        self.STATE_HEIGHT = stateShape[1]

        # Defining the input variables for training and evaluation.
        self.stacked_frames = Input(shape=(1, self.STATE_WIDTH, self.STATE_HEIGHT))
        self.action = Input(shape=(self.num_actions,))
        self.R = Input(shape=(1,))
        self.v_calc = Input(shape=(1,))
                
        self.pi = self.build_model()
        self.v = self.build_model()
        
    def build_model(self):
        model = Sequential()

        model.add(Conv2D(16, (8, 8), activation='sigmoid', strides=4, input_shape=(self.STATE_WIDTH, self.STATE_HEIGHT, 3)))
        model.add(Conv2D(32, (4, 4), activation='sigmoid', strides=2))

        model.add(Dense(units=64, activation='sigmoid', input_dim=100))
        model.add(Dense(units=10, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        return model
        
    def train(self, states, actions, Rs, calc_diff):

        diff = None
        
        if calc_diff:
            # Save the parameters before a training step.
            self.update_pi = []
            for x in self.pms_pi:
                self.update_pi.append(x.value)
            self.update_v = []
            for x in self.pms_v:
                self.update_v.append(x.value)

        actions_1hot = []
        for action in actions:
            # Training part
            action_as_array = np.zeros(self.num_actions, dtype=np.float32)
            action_as_array[int(action)] = 1
            actions_1hot.append(action_as_array)
        
        v_calcs = []
        for state in states:
            v_calcs.append(self.state_value(state))

        float32_Rs = np.float32(Rs) # Without this, CNTK warns to use float32 instead of float64 to enhance performance.

        self.pi.fit({self.stacked_frames: states, self.action: actions_1hot, self.R: float32_Rs, self.v_calc: v_calcs})
        self.v.fit({self.stacked_frames: states, self.R: float32_Rs})
        
    # Called
    def state_value(self, state):
        return self.v.predict({self.stacked_frames: [state]}, batch_size=None, verbose=0, steps=None)
    
    # Called by Agent
    def pi_probabilities(self, state):
        return self.pi.predict({self.stacked_frames: [state]}, batch_size=None, verbose=0, steps=None)
    
    # Called by Agent
    def get_num_actions(self):
        return self.num_actions
        
    # Called for logging
    def get_last_avg_loss(self):
        return self.trainer_pi.previous_minibatch_loss_average + self.trainer_v.previous_minibatch_loss_average
            
    # Called through logger 
    def load_model(self, file_name_pi, file_name_v):
        self.pi = load_model(file_name_pi) # load(fn) does different things, it would
        self.v = load_model(file_name_v)   # create a new function. It did not work.
        
    # Called
    def save_model(self, file_name_pi, file_name_v):
        self.pi.save(file_name_pi)
        self.v.save(file_name_v)