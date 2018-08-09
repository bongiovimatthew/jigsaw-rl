import numpy as np
import cntk
from cntk.device import try_set_default_device, cpu
from cntk.layers import Convolution2D, Dense, Sequential, BatchNormalization
from cntk.learners import adam, learning_rate_schedule, momentum_schedule, UnitType
from cntk.logging import TensorBoardProgressWriter
from celery.contrib import rdb
import random as r

class DeepNet:
    
    def __init__(self, num_actions, lr):

        self.num_actions = num_actions
        self.lr = lr
        
        self.build_model()
        self.build_trainer()

        
    def build_model(self):
        
        # Defining the input variables for training and evaluation.
        self.stacked_frames = cntk.input_variable((1, 84, 84), dtype=np.float32)
        self.action = cntk.input_variable(self.num_actions)
        self.R = cntk.input_variable(1, dtype=np.float32)
        self.v_calc = cntk.input_variable(1, dtype=np.float32) # In the loss of pi, the parameters of V(s) should be fixed.
        
        # Creating the value approximator extension.
        conv1_v = Convolution2D((8, 8), num_filters = 16, pad = False, strides=4, activation=cntk.relu, name='conv1_v')
        conv2_v = Convolution2D((4, 4), num_filters = 32, pad = False, strides=2, activation=cntk.relu, name='conv2_v')
        dense_v = Dense(256, activation=cntk.relu, name='dense_v')
        v = Sequential([conv1_v, conv2_v, dense_v, Dense(1, activation=cntk.relu, name='outdense_v')])
        
        # Creating the policy approximator extension.
        conv1_pi = Convolution2D((8, 8), num_filters = 16, pad = False, strides=4, activation=cntk.relu, name='conv1_pi')
        conv2_pi = Convolution2D((4, 4), num_filters = 32, pad = False, strides=2, activation=cntk.relu, name='conv2_pi')
        dense_pi = Dense(256, activation=cntk.relu, name='dense_pi')
        pi = Sequential([conv1_pi, conv2_pi, dense_pi, Dense(self.num_actions, activation=cntk.softmax, name='outdense_pi')])
        
        self.pi = pi(self.stacked_frames)
        self.pms_pi = self.pi.parameters # List of cntk Parameter types (containes the function's parameters)
        self.v = v(self.stacked_frames)
        self.pms_v = self.v.parameters
        
        display_model(pi)
        display_model(v)
        cntk.debugging.debug_model(v)
        # action_as_array   

    def build_trainer(self):
                
        # Set the learning rate, and the momentum parameters for the Adam optimizer.
        lr = learning_rate_schedule(self.lr, UnitType.minibatch)
        beta1 = momentum_schedule(0.9)
        beta2 = momentum_schedule(0.99)
        
        # Calculate the losses.
        loss_on_v = cntk.squared_error(self.R, self.v)
        pi_a_s = cntk.log(cntk.times_transpose(self.pi, self.action))

        #loss_on_pi = cntk.variables.Constant(-1) * (cntk.plus(cntk.times(pi_a_s, cntk.minus(self.R, self.v_calc)), 0.01 * cntk.times_transpose(self.pi, cntk.log(self.pi))))
        loss_on_pi = cntk.times(pi_a_s, cntk.minus(self.R, self.v_calc))
        
        self.tensorboard_v_writer = TensorBoardProgressWriter(freq=10, log_dir="tensorboard_v_logs", model=self.v)
        self.tensorboard_pi_writer = TensorBoardProgressWriter(freq=10, log_dir="tensorboard_pi_logs", model=self.pi)

        # tensorboard --logdir=log  http://localhost:6006/ 

        # Create the trainiers.
        trainer_v = cntk.Trainer(self.v, (loss_on_v), [adam(self.pms_v, lr, beta1, variance_momentum=beta2, gradient_clipping_threshold_per_sample = 2, l2_regularization_weight=0.01)], self.tensorboard_v_writer)
        trainer_pi = cntk.Trainer(self.pi, (loss_on_pi), [adam(self.pms_pi, lr, beta1, variance_momentum=beta2, gradient_clipping_threshold_per_sample = 2, l2_regularization_weight=0.01)], self.tensorboard_pi_writer)
    
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

        trained_pi = self.trainer_pi.train_minibatch({self.stacked_frames: states, self.action: actions_1hot, self.R: float32_Rs, self.v_calc: v_calcs})
        trained_v = self.trainer_v.train_minibatch({self.stacked_frames: states, self.R: float32_Rs})

        print("v_calc:{0} float32_R:{1} action:{2} trained_pi:{3} trained_v:{4}".format(v_calcs[0], float32_Rs[0], actions[0], trained_pi, trained_v))
        
        if calc_diff:
            # Calculate the differences between the updated and the original params.
            for idx in range(len(self.pms_pi)):
                self.tb_pi.write_value(self.pms_pi[idx].uid + "/mean",  reduce_mean(self.pms_pi[idx]).eval(), idx)
                self.update_pi[idx] = self.pms_pi[idx].value - self.update_pi[idx]
            for idx in range(len(self.pms_v)):
                self.update_v[idx] = self.pms_v[idx].value - self.update_v[idx]
            
            diff = [self.update_pi, self.update_v]
        
        return diff
        
    def state_value(self, state):
        print("v.val:",self.v.eval(state))
        return self.v.eval({self.stacked_frames: [state]})
    
    def pi_probabilities(self, state):
        return self.pi.eval({self.stacked_frames: [state]})
    
    def get_num_actions(self):
        return self.num_actions
        
    def get_last_avg_loss(self):
        return self.trainer_pi.previous_minibatch_loss_average + self.trainer_v.previous_minibatch_loss_average
    
    def synchronize_net(self, shared): 
        for idx in range(0, len(self.pms_pi)):
            self.pms_pi[idx].value = shared[0][idx]
        for idx in range(0, len(self.pms_v)):
            self.pms_v[idx].value = shared[1][idx]
                    
    def sync_update(self, shared, diff):
        for idx in range(0, len(self.pms_pi)):
            shared[0][idx] += diff[0][idx]
        for idx in range(0, len(self.pms_v)):
            shared[1][idx] += diff[1][idx]
    
    def get_parameters_pi(self):
        pickle_prms_pi = []
        for x in self.pms_pi:
            pickle_prms_pi.append(x.value) # .value gives numpy arrays
        return pickle_prms_pi
        
    def get_parameters_v(self):
        pickle_prms_v = []
        for x in self.pms_v:
            pickle_prms_v.append(x.value) # .value gives numpy arrays
        return pickle_prms_v              # to avoid: can't pickle SwigPyObject
        
    def load_model(self, file_name_pi, file_name_v):
        self.pi.restore(file_name_pi) # load(fn) does different things, it would
        self.v.restore(file_name_v)   # create a new function. It did not work.
        
    def save_model(self, file_name_pi, file_name_v):
        self.pi.save(file_name_pi)
        self.v.save(file_name_v)
       
class DnnAgent:
    
    def action(net, state): # Take into account None as input -> generate random actions
        
        act = 0
        n = net.get_num_actions()
        if state is None:
            act = r.randint(0, n-1) 
        else:
            # Decide to explore or not. (In order to avoid the moveless situations.)
            explore = r.randint(0, 1000)
            if explore < 0.05 * 1000:
                act = r.randint(0, n-1)
            else:
                prob_vec = net.pi_probabilities(state)[0] * 1000
                candidate = r.randint(0, 1000)
            
                for i in range(0, n):
                    if prob_vec[i] >= candidate:
                        act = i
        
        return act
        
    def action_with_exploration(net, state, epsilon): # Take into account None as input -> generate random actions
                                                      # Epsilon-greedy is a right approach.
        act = 0
        n = net.get_num_actions()
        if state is None:
            act = r.randint(0, n-1) 
        else:
            # Decide to explore or not.
            explore = r.randint(0, 1000)
            if explore < epsilon * 1000:
                act = r.randint(0, n-1)
            else:
                prob_vec = net.pi_probabilities(state)[0] * 1000

                candidate = r.randint(0, 1000)
            
                for i in range(0, n):
                    if prob_vec[i] >= candidate:
                        act = i
        
        return act