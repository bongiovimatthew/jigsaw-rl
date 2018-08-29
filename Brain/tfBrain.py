import tensorflow as tf
import numpy as np
from collections import namedtuple

from Brain.IBrain import IBrain 

tf_ops = namedtuple('tf_ops', 'loss_op accuracy_op train_op prediction_op')


# class BaseTFModel():

# class Critic(BaseTFModel):

# class Actor():

class TFBrain(IBrain): 

    PI_MODEL = 1
    V_MODEL = 2

    def __init__(self, num_actions, lr, stateShape):

        self.num_actions = num_actions
        self.lr = lr
        self.debugMode = True
        self.STATE_WIDTH = stateShape[0]
        self.STATE_HEIGHT = stateShape[1]

        # Defining the input variables for training and evaluation.
        self.state = tf.placeholder(tf.float32, [None, 1, self.STATE_WIDTH, self.STATE_HEIGHT], "State")
        self.action = tf.placeholder(tf.int32, [None], "Action")
        self.R = tf.placeholder(tf.float32, [None], "RewardR")
        self.v_calc = tf.placeholder(tf.float32, [None], "V_Calc")
                
        # self.pi_model = self.build_pi_model()
        # self.v_model = self.build_v_model()
        self.dropout = 0.01

        self.init_critic()
        self.init_actor()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        print("learning_rate: ", self.lr)
        # # Store layers weight & bias
        # self.weights = {
        #     # 5x5 conv, 1 input, 32 outputs
        #     'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        #     # 5x5 conv, 32 inputs, 64 outputs
        #     'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        #     # fully connected, 7*7*64 inputs, 1024 outputs
        #     'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
        #     # 1024 inputs, 10 outputs (class prediction)
        #     'out': tf.Variable(tf.random_normal([1024, num_classes]))
        # }

        # self.biases = {
        #     'bc1': tf.Variable(tf.random_normal([32])),
        #     'bc2': tf.Variable(tf.random_normal([64])),
        #     'bd1': tf.Variable(tf.random_normal([1024])),
        #     'out': tf.Variable(tf.random_normal([num_classes]))
        # }


    def create_model(self, dropout, reuse, is_training, modelToCreate):
        modelName = ""

        if modelToCreate == TFBrain.PI_MODEL:
            modelName = "PiModel"
        elif modelToCreate == TFBrain.V_MODEL:
            modelName = "VModel"
        else:
            raise Exception("Invalid model ID passed")

        # Define a scope for reusing the variables
        with tf.variable_scope(modelName, reuse=reuse):
            # TF Estimator input is a dict, in case of multiple inputs
            # x = x_dict # ['images']

            # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
            # Reshape to match picture format [Height x Width x Channel]
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            local_state = tf.reshape(self.state, shape=[-1, self.STATE_WIDTH, self.STATE_HEIGHT, 1])

            # Convolution Layer with no padding
            conv1 = tf.layers.conv2d(local_state, filters=16, kernel_size=(8, 8), strides=4, padding="VALID", use_bias=True, activation=tf.nn.sigmoid)

            # # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            # conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=(4, 4), strides=2, padding="VALID", use_bias=True, activation=tf.nn.sigmoid)

            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            # conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv2)

            # Fully connected layer (in tf contrib folder for now)
            fc1 = tf.contrib.layers.fully_connected(fc1, 256, activation_fn=tf.nn.sigmoid)

            # Apply Dropout (if is_training is False, dropout is not applied)
            # REMOVING DROPOUT TEMPORARILY
            # fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)


            if modelToCreate == TFBrain.PI_MODEL:
                out = tf.contrib.layers.fully_connected(fc1, self.num_actions, activation_fn=tf.nn.softmax)

            else:
                out = tf.contrib.layers.fully_connected(fc1, 1, activation_fn=tf.nn.sigmoid)


            # add_summaries = not reuse
            # if add_summaries:
            #     tf.contrib.layers.summarize_activation(conv1)
            #     tf.contrib.layers.summarize_activation(conv2)
            #     tf.contrib.layers.summarize_activation(fc1)
            #     tf.contrib.layers.summarize_activation(out)


        return out

    # # Create the neural network for the actor
    # def build_pi_model(x_dict, dropout, reuse, is_training):
    #     return create_model(x_dict, dropout, reuse, is_training, PI_MODEL)

    def init_critic(self):
        # Build the neural network
        # Because Dropout have different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.

        logits_train = (self.create_model(self.dropout, reuse = False, is_training = True, modelToCreate = TFBrain.V_MODEL))#,  squeeze_dims=[1])
        logits_test =  (self.create_model(self.dropout, reuse = True, is_training = False, modelToCreate = TFBrain.V_MODEL))#,  squeeze_dims=[1])

        # feed_dict={self.state: state}

        # # Predictions
        # pred_classes = tf.argmax(logits_test, axis=1)
        # pred_probas = tf.nn.softmax(logits_test)

        # If prediction mode, early return
        # if mode == tf.estimator.ModeKeys.PREDICT:
        #     return sess.run(logits_test, feed_dict=feed_dict)
            # return tf.estimator.EstimatorSpec(mode, predictions=logits_test)

            # Define loss and optimizer
        # loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
        loss_op = tf.reduce_mean(tf.squared_difference(logits_train, self.R))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())

        print("global_step: ", tf.train.get_global_step())
        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=self.R, predictions=logits_test)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        # estim_specs = tf.estimator.EstimatorSpec(
        #     mode=mode,
        #     predictions=logits_test,
        #     loss=loss_op,
        #     train_op=train_op,
        #     eval_metric_ops={'accuracy': acc_op})

        self.critic_ops = tf_ops(loss_op = loss_op, accuracy_op = acc_op, train_op = train_op, prediction_op = logits_test)

        return

    # Define the model function (following TF Estimator Template)
    # Create the neural network for the critic
    # def build_v_model():
    #     model = tf.estimator.Estimator(v_model_fn)

    #     return model

    def init_actor(self):
        # Build the neural network
        # Because Dropout have different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.

        logits_train = self.create_model(self.dropout, reuse = False, is_training = True, modelToCreate = TFBrain.PI_MODEL)
        logits_test = self.create_model(self.dropout, reuse = True, is_training = False, modelToCreate = TFBrain.PI_MODEL)

        probs_fixed = logits_train + 1e-8
        test_probs_fixed = logits_test + 1e-8

        batch_size = tf.shape(self.state)[0]

        # # Predictions
        # pred_classes = tf.argmax(logits_test, axis=1)
        # pred_probas = tf.nn.softmax(logits_test)

        # If prediction mode, early return
        # if mode == tf.estimator.ModeKeys.PREDICT:
        #     return tf.estimator.EstimatorSpec(mode, predictions=logits_test)

            # Define loss and optimizer
        # loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))

        # -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)

        # responsible_outputs = tf.reduce_sum(logits_test * self.actions_onehot, [1])

        gather_indices = tf.range(batch_size) * tf.shape(probs_fixed)[1] + self.action
        picked_action_probs = tf.gather(tf.reshape(probs_fixed, [-1]), gather_indices)

        entropy = -tf.reduce_sum(probs_fixed * tf.log(probs_fixed), 1, name="entropy")
        loss_op = tf.reduce_sum(-(tf.log(picked_action_probs) * (self.R - self.v_calc) + 0.01 * entropy))

        # loss_op = -tf.reduce_sum(tf.log(responsible_outputs)*self.advantages)
        # loss_op = cntk.plus(tf.multiply(tf.constant(-1), cntk.times(pi_a_s, cntk.minus(self.R, self.v_calc)), 0.01 * cntk.times_transpose(self.pi, cntk.log(self.pi)))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        # acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        # estim_specs = tf.estimator.EstimatorSpec(
        #     mode=mode,
        #     predictions=logits_test,
        #     loss=loss_op,
        #     train_op=train_op,
        #     eval_metric_ops={'accuracy': acc_op})

        # return estim_specs
        self.actor_ops = tf_ops(loss_op = loss_op, accuracy_op = None, train_op = train_op, prediction_op = test_probs_fixed)

        return



    # Define the model function (following TF Estimator Template)
    # Create the neural network for the critic
    # def build_pi_model():
    #     model = tf.estimator.Estimator(pi_model_fn)

    #     return model

        
    def train(self, states, actions, Rs, calc_diff):
        diff = None
        # actions_1hot = []
        # for action in actions:
        #     # Training part
        #     action_as_array = np.zeros(self.num_actions, dtype=np.float32)
        #     action_as_array[int(action)] = 1
        #     actions_1hot.append(action_as_array)
        
        # v_calcs = []
        # for state in states:
        #     v_calcs.append(self.state_value(state))

        # float32_Rs = np.float32(Rs) 

        # v_input_fn = tf.estimator.inputs.numpy_input_fn(
        #     x=states, y=float32_Rs,
        #     batch_size=len(float32_Rs), num_epochs=None, shuffle=True)


        # self.v_model.train(v_input_fn)

        # We should split the 2 models into 2 classes to prevent this
        fixedUpRs = np.array(Rs).flatten()
        v_feed_dict = {self.state: states, self.action: (actions), self.R: (fixedUpRs), self.v_calc: fixedUpRs}

        v_calcs = np.array(self.sess.run(self.critic_ops.prediction_op, v_feed_dict)).flatten()

        if self.debugMode:
            print("v_calcs: ", v_calcs)

        print(np.array(Rs).flatten().shape)

        pi_feed_dict = {self.state: states, self.action: (actions), self.R: (fixedUpRs), self.v_calc: v_calcs}
        # print(len(Rs))
        # print(len(Rs[0]))
        # print(len(Rs))
        # print(len(actions))
        # print(actions.shape)
        # print(Rs.shape)

        self.sess.run(
            [
            self.actor_ops.loss_op,
            self.actor_ops.train_op,
            ], 
            pi_feed_dict)

        self.sess.run(
            [
            self.critic_ops.loss_op,
            self.critic_ops.train_op,
            ], 
            v_feed_dict)


        # if calc_diff:
        #     # Save the parameters before a training step.
        #     self.update_pi = []
        #     for x in self.pms_pi:
        #         self.update_pi.append(x.value)
        #     self.update_v = []
        #     for x in self.pms_v:
        #         self.update_v.append(x.value)

        # self.pi.fit({self.stacked_frames: states, self.action: actions_1hot, self.R: float32_Rs, self.v_calc: v_calcs})
        # self.v.fit({self.stacked_frames: states, self.R: float32_Rs})
        
    # Called
    def state_value(self, state):
        feed_dict = {self.state: [state]}
        return self.sess.run(self.critic_ops.prediction_op, feed_dict)
    
    # Called by Agent
    def pi_probabilities(self, state):
        feed_dict = {self.state: [state]}
        probs_to_return = self.sess.run(self.actor_ops.prediction_op, feed_dict)
        print("SHAPE ", probs_to_return.shape)
        return probs_to_return

    def action_probabilities(self, state):
        return self.pi_probabilities(state)[0]

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

    # def __del__(self):
    #     print("Closing tf session")
    #     self.sess.close()