from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D
from keras.models import load_model
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
from keras import backend as K

from Brain.IBrain import IBrain

import numpy as np


class KerasNetBrain(IBrain):

    def __init__(self, num_actions, lr, stateShape):

        self.num_actions = num_actions
        self.lr = lr
        self.debugMode = True
        self.STATE_WIDTH = stateShape[0]
        self.STATE_HEIGHT = stateShape[1]

        # Defining the input variables for training and evaluation.

        if K.image_data_format() == 'channels_first':
            input_shape = (1, self.STATE_WIDTH, self.STATE_HEIGHT)
        else:
            input_shape = (self.STATE_WIDTH, self.STATE_HEIGHT, 1)

        self.input_shape = input_shape

        self.state_image = Input(shape=input_shape, name="state_image_input", dtype='float32')
        self.action = Input(shape=(self.num_actions,), name="action_input")
        self.R = Input(shape=(1,), name="reward_input", dtype='float32')
        self.v_calc = Input(shape=(1,), name="value_function_input", dtype='float32')

        #self.pi = self.build_model('pi')
        #self.v = self.build_model('v')

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()
        # This placeholder is used to feed dError / dCritic (from critic model)
        self.actor_critic_grad = K.placeholder(dtype='float32', [None, self.input_shape])

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights, -self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = keras.optimizers.Adam(
            lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False).apply_gradients(grads)

        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()
        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    def create_critic_model(self):
        state_input = Input(shape=self.input_shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.num_actions)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=self.lr)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def create_actor_model(self):
        state_input = Input(shape=self.input_shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.num_actions, activation='softmax')(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=self.lr)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def build_model(self, type):

        loss_on_v = cntk.squared_error(self.R, self.v)
        pi_a_s = cntk.log(cntk.times_transpose(self.pi, self.action))

        loss_on_pi = cntk.variables.Constant(-1) * (cntk.plus(cntk.times(pi_a_s, cntk.minus(
            self.R, self.v_calc)), 0.01 * cntk.times_transpose(self.pi, cntk.log(self.pi))))

    # def build_model(self, type):
    #     model = Sequential()

    #     if K.image_data_format() == 'channels_first':
    #         input_shape = (1, self.STATE_WIDTH, self.STATE_HEIGHT)
    #     else:
    #         input_shape = (self.STATE_WIDTH, self.STATE_HEIGHT, 1)

    #     print(input_shape)
    #     print("image data format")
    #     print(K.image_data_format())

    #     model.add(Conv2D(16, (8, 8), activation='sigmoid', strides=4, input_shape=input_shape))
    #     model.add(Conv2D(32, (4, 4), activation='sigmoid', strides=2))
    #     model.add(Dense(256, activation='sigmoid'))

    #     if type == 'pi':
    #         model.add(Dense(self.num_actions, activation='softmax'))
    #     else:
    #         model.add(Dense(1, activation='sigmoid'))

    #     model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    #     return model

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

        # Without this, CNTK warns to use float32 instead of float64 to enhance performance.
        float32_Rs = np.float32(Rs)

        self.pi.fit({'state_image_input': states[0], 'action_input': actions_1hot[0],
                     'reward_input': float32_Rs[0], 'value_function_input': v_calcs[0]})
        self.v.fit({'state_image_input': states, 'reward_input': float32_Rs})

    def process_state(self, state):
        if K.image_data_format() == 'channels_first':
            img_final = state
        else:
            img_final = np.reshape(state, (self.STATE_WIDTH, self.STATE_HEIGHT, 1))

        return np.array([img_final])

    # Called
    def state_value(self, state):
        proc_state = self.process_state(state)
        return self.v.predict(proc_state)

    # Called by Agent
    def pi_probabilities(self, state):
        proc_state = self.process_state(state)
        return self.pi.predict(proc_state)

    # Called by Agent
    def get_num_actions(self):
        return self.num_actions

    # Called for logging
    def get_last_avg_loss(self):
        return self.trainer_pi.previous_minibatch_loss_average + self.trainer_v.previous_minibatch_loss_average

    # Called through logger
    def load_model(self, file_name_pi, file_name_v):
        self.pi = load_model(file_name_pi)  # load(fn) does different things, it would
        self.v = load_model(file_name_v)   # create a new function. It did not work.

    # Called
    def save_model(self, file_name_pi, file_name_v):
        self.pi.save(file_name_pi)
        self.v.save(file_name_v)
