import numpy as np
import cntk
from cntk.device import try_set_default_device, cpu
from cntk.layers import Convolution2D, Dense, Sequential, BatchNormalization, MaxPooling
from cntk.learners import adam, learning_rate_schedule, momentum_schedule, UnitType
import pdb
from PIL import Image
from IPython.display import SVG, display

def display_model(model):
    return
    # svg = cntk.logging.graph.plot(model, "tmp.svg")
    # display(SVG(filename="tmp.svg"))

# Set CPU as device for the neural network.
try_set_default_device(cpu())

class DeepNet:
    
    def __init__(self, num_actions, lr):
        self.num_actions = num_actions
        self.lr = lr
        self.debugMode = True
        self.num_steps = 0

        self.build_model()
        self.build_trainer()
        
    def build_model(self):
        
        cntk.debugging.set_checked_mode(True)
        # Defining the input variables for training and evaluation.
        self.stacked_frames = cntk.input_variable((1, 168, 168), dtype=np.float32)
        self.action = cntk.input_variable(self.num_actions)
        self.R = cntk.input_variable(1, dtype=np.float32)
        self.v_calc = cntk.input_variable(1, dtype=np.float32) # In the loss of pi, the parameters of V(s) should be fixed.
        
        # Creating the value approximator extension.
        conv1_v = Convolution2D((8, 8), num_filters = 16, pad = False, strides=4, activation=cntk.sigmoid, name='conv1_v')
        conv2_v = Convolution2D((4, 4), num_filters = 32, pad = False, strides=2, activation=cntk.sigmoid, name='conv2_v')
        # pooling_v = MaxPooling((4,4), 4, name='pooling_v')
        dense_v = Dense(256, activation=cntk.sigmoid, name='dense_v')
        # cntk.debugging.set_computation_network_trace_level(1)
        v = Sequential([conv1_v, conv2_v, dense_v, Dense(1, activation=cntk.sigmoid, name='outdense_v')])
        
        # Creating the policy approximator extension.
        conv1_pi = Convolution2D((8, 8), num_filters = 16, pad = False, strides=4, activation=cntk.sigmoid, name='conv1_pi')
        conv2_pi = Convolution2D((4, 4), num_filters = 32, pad = False, strides=2, activation=cntk.sigmoid, name='conv2_pi')
        # pooling_pi = MaxPooling((4,4), 4, name='pooling_pi')
        dense_pi = Dense(256, activation=cntk.sigmoid, name='dense_pi')
        pi = Sequential([conv1_pi, conv2_pi, dense_pi, Dense(self.num_actions, activation=cntk.softmax, name='outdense_pi')])
        # pdb.set_trace()
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
        loss_on_pi = cntk.times(cntk.variables.Constant(-1) * pi_a_s, cntk.minus(self.R, self.v_calc))
        
        # Create the trainiers.
        trainer_v = cntk.Trainer(self.v, (loss_on_v), [adam(self.pms_v, lr, beta1, variance_momentum=beta2, gradient_clipping_threshold_per_sample = 2, l2_regularization_weight=0.01)])
        trainer_pi = cntk.Trainer(self.pi, (loss_on_pi), [adam(self.pms_pi, lr, beta1, variance_momentum=beta2, gradient_clipping_threshold_per_sample = 2, l2_regularization_weight=0.01)])
        
        self.trainer_pi = trainer_pi 
        self.trainer_v = trainer_v

    def combineImage(self, layer, layer_output):
        # return
        # print(layer_output.shape)
        # print(layer.shape)

        # np.swapaxes(layer_output, 0, 1)
        # np.swapaxes(layer_output, 1, 2)

        layer_output = layer_output[0]
        imageArray = np.zeros((layer_output.shape[1], layer_output.shape[2]))

        for layer_index in range(layer_output.shape[0]):
            imageArray += layer_output[layer_index]

        imageArray = np.uint8(imageArray / layer_output.shape[0])

        # print(imageArray)
        imToShow = Image.fromarray(imageArray, 'L')
        imToShow.show()


        l
    
    def train_net(self, states, actions, Rs, calc_diff):
        self.num_steps += 1
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
        # print("v_calc:",v_calc)
        # rdb.set_trace()
        float32_Rs = np.float32(Rs) # Without this, CNTK warns to use float32 instead of float64 to enhance performance.
        
        # print(R)
        # print("v_calc:{0} float32_R:{1}".format(v_calc, float32_R))

        self.trainer_pi.train_minibatch({self.stacked_frames: states, self.action: actions_1hot, self.R: float32_Rs, self.v_calc: v_calcs})
        self.trainer_v.train_minibatch({self.stacked_frames: states, self.R: float32_Rs})
            # net.pi.

        # if self.debugMode and 0 == (self.num_steps % 50):
        # conv1_v = cntk.combine([self.pi.find_by_name('conv2_v').owner])
        # print(conv1_v)
        #     conv1_v = cntk.combine([self.pi.find_by_name('conv1_v').owner])
        #     conv2_v = cntk.combine([self.pi.find_by_name('conv2_v').owner])
        #     self.combineImage(conv2_v, conv2_v.eval(state))
        
        if calc_diff:
            # Calculate the differences between the updated and the original params.
            for idx in range(len(self.pms_pi)):
                self.update_pi[idx] = self.pms_pi[idx].value - self.update_pi[idx]
            for idx in range(len(self.pms_v)):
                self.update_v[idx] = self.pms_v[idx].value - self.update_v[idx]
            
            diff = [self.update_pi, self.update_v]
        
        return diff
        
    def state_value(self, state):
        # print(state)
        # print(self.v.eval({self.stacked_frames: [state]}))
        return self.v.eval({self.stacked_frames: [state]})
    
    def pi_probabilities(self, state):
        # print((self.pi.eval({self.stacked_frames: [state]})).shape)
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
        
# Functions to generate the next actions
import random as r
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
        # print(epsilon)

        explore = np.random.binomial(1, epsilon)
        # explore = r.randint(0, 1000)
        if explore:
            act = r.randint(0, n-1)
            # print("Explored")
        else:
            prob_vec = net.pi_probabilities(state)[0]
            maxProbability = prob_vec[0]
            possibleActions = [0]

            for i in range(1, n):
                if prob_vec[i] > maxProbability:
                    maxProbability = prob_vec[i]
                    possibleActions = [i]
                elif prob_vec[i] == maxProbability:
                    possibleActions.append(i)

            act = possibleActions[np.random.randint(0, len(possibleActions))]

            print("Exploited: :{0}, act: {1}".format(prob_vec, act))


    return act
