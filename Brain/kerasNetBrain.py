from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D

class KerasNet: 
	
	def __init__(self, num_actions, lr, stateShape):

        self.num_actions = num_actions
        self.lr = lr
        self.debugMode = True
        self.STATE_WIDTH = stateShape[0]
        self.STATE_HEIGHT = stateShape[1]
        
        self.model = self.build_model()
        self.build_trainer()
        
    def build_model(self):
        model = Sequential()

        model.add(Conv2D(16, (8, 8), activation='sigmoid', strides=4, input_shape=(self.STATE_WIDTH, self.STATE_HEIGHT, 3)))
        model.add(Conv2D(32, (4, 4), activation='sigmoid', strides=2))

 		model.add(Dense(units=64, activation='sigmoid', input_dim=100))
		model.add(Dense(units=10, activation='softmax'))

		return model
        
    def build_trainer(self):
        return model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        
    # Called
    def train_net(self, states, actions, Rs, calc_diff):

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

        model.fit(x_train, y_train, epochs=5, batch_size=32)
        
        trained_pi = self.trainer_pi.train_minibatch({self.stacked_frames: states, self.action: actions_1hot, self.R: float32_Rs, self.v_calc: v_calcs})
        trained_v = self.trainer_v.train_minibatch({self.stacked_frames: states, self.R: float32_Rs})
        
        if calc_diff:
            # Calculate the differences between the updated and the original params.
            for idx in range(len(self.pms_pi)):
                self.tb_pi.write_value(self.pms_pi[idx].uid + "/mean",  reduce_mean(self.pms_pi[idx]).eval(), idx)
                self.update_pi[idx] = self.pms_pi[idx].value - self.update_pi[idx]
            for idx in range(len(self.pms_v)):
                self.update_v[idx] = self.pms_v[idx].value - self.update_v[idx]
            
            diff = [self.update_pi, self.update_v]
        
        return diff
        
    # Called
    def state_value(self, state):
        return self.v.eval({self.stacked_frames: [state]})
    
    # Called by Agent
    def pi_probabilities(self, state):
        return self.pi.eval({self.stacked_frames: [state]})
    
    # Called by Agent
    def get_num_actions(self):
        return self.num_actions
        
    # Called for logging
    def get_last_avg_loss(self):
        return self.trainer_pi.previous_minibatch_loss_average + self.trainer_v.previous_minibatch_loss_average
            
    # Called through logger 
    def load_model(self, file_name_pi, file_name_v):
        self.pi.restore(file_name_pi) # load(fn) does different things, it would
        self.v.restore(file_name_v)   # create a new function. It did not work.
        
    # Called
    def save_model(self, file_name_pi, file_name_v):
        self.pi.save(file_name_pi)
        self.v.save(file_name_v)