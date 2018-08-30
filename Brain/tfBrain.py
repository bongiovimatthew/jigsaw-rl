import tensorflow as tf
import numpy as np

from Brain.IBrain import IBrain 


class BaseTFModel():
    def create_base_model(self, dropout, reuse, is_training):

        local_state = tf.reshape(self.state, shape=[-1, self.STATE_WIDTH, self.STATE_HEIGHT, 1])

        # Convolution Layer with no padding
        conv1 = tf.layers.conv2d(local_state, filters=16, kernel_size=(8, 8), strides=4, padding="VALID", use_bias=True, activation=tf.nn.sigmoid)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=(4, 4), strides=2, padding="VALID", use_bias=True, activation=tf.nn.sigmoid)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.contrib.layers.fully_connected(fc1, 256, activation_fn=tf.nn.sigmoid)

        # Apply Dropout (if is_training is False, dropout is not applied)
        # REMOVING DROPOUT TEMPORARILY
        # fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        add_summaries = not reuse
        if add_summaries:
            tf.contrib.layers.summarize_activation(conv1)
            tf.contrib.layers.summarize_activation(conv2)
            tf.contrib.layers.summarize_activation(fc1)

        return fc1


class Critic(BaseTFModel):
    def __init__(self, num_actions, lr, stateShape):

        self.num_actions = num_actions
        self.lr = lr
        self.debugMode = True
        self.STATE_WIDTH = stateShape[0]
        self.STATE_HEIGHT = stateShape[1]

        # Defining the input variables for training and evaluation.
        self.state = tf.placeholder(tf.float32, [None, 1, self.STATE_WIDTH, self.STATE_HEIGHT], "State")
        self.R = tf.placeholder(tf.float32, [None], "RewardR")
                
        self.dropout = 0.01
     
        with tf.variable_scope("CriticModel"):
            # Create the train and test graphs
            logits_train = self.create_model(self.dropout, reuse = False, is_training = True)
            logits_test = self.create_model(self.dropout, reuse = True, is_training = False)


            self.loss_op = tf.reduce_sum(tf.squared_difference(logits_train, self.R), name="Critic_loss")

            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(self.loss_op,
                                          global_step=tf.train.get_global_step())

            # Evaluate the accuracy of the model
            # self.acc_op = tf.metrics.accuracy(labels=self.R, predictions=logits_test)

            self.prediction_op = logits_test


            # Summaries
            prefix = tf.get_variable_scope().name
            tf.summary.scalar(self.loss_op.name, self.loss_op)
            tf.summary.scalar("{}/max_value".format(prefix), tf.reduce_max(logits_train))
            tf.summary.scalar("{}/min_value".format(prefix), tf.reduce_min(logits_train))
            tf.summary.scalar("{}/mean_value".format(prefix), tf.reduce_mean(logits_train))
            tf.summary.scalar("{}/reward_max".format(prefix), tf.reduce_max(self.R))
            tf.summary.scalar("{}/reward_min".format(prefix), tf.reduce_min(self.R))
            tf.summary.scalar("{}/reward_mean".format(prefix), tf.reduce_mean(self.R))
            tf.summary.histogram("{}/reward_targets".format(prefix), self.R)
            tf.summary.histogram("{}/values".format(prefix), logits_train)


        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries = [s for s in summary_ops if "CriticModel" in s.name]

        self.summaries = tf.summary.merge(summaries)

        return

    def create_model(self, dropout, reuse, is_training):
        with tf.variable_scope("CriticModel", reuse=reuse):
            fc1 = self.create_base_model(self.dropout, reuse, is_training)
            out = tf.contrib.layers.fully_connected(fc1, 1, activation_fn=tf.nn.sigmoid)

            add_summaries = not reuse
            if add_summaries:
                tf.contrib.layers.summarize_activation(out)

        return out

class Actor(BaseTFModel):
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
                
        self.dropout = 0.01
   
        with tf.variable_scope("ActorModel"):
            logits_train = self.create_model(self.dropout, reuse = False, is_training = True)
            logits_test = self.create_model(self.dropout, reuse = True, is_training = False)

            probs_fixed = logits_train + 1e-8
            test_probs_fixed = logits_test + 1e-8

            batch_size = tf.shape(self.state)[0]

            gather_indices = tf.range(batch_size) * tf.shape(probs_fixed)[1] + self.action
            picked_action_probs = tf.gather(tf.reshape(probs_fixed, [-1]), gather_indices)

            entropy = -tf.reduce_sum(probs_fixed * tf.log(probs_fixed), 1, name="actor_entropy")
            self.loss_op = tf.reduce_sum((-(tf.log(picked_action_probs) * (self.R - self.v_calc) + 0.01 * entropy)), name="actor_loss")


            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(self.loss_op,
                                          global_step=tf.train.get_global_step())

            self.prediction_op = test_probs_fixed


            tf.summary.scalar(self.loss_op.name, self.loss_op)
            tf.summary.histogram(entropy.op.name, entropy)

        
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries = [s for s in summary_ops if "ActorModel" in s.name]

        self.summaries = tf.summary.merge(summaries)

        return

    def create_model(self, dropout, reuse, is_training):        
        with tf.variable_scope("ActorModel", reuse=reuse):
            fc1 = self.create_base_model(self.dropout, reuse, is_training)
            out = tf.contrib.layers.fully_connected(fc1, self.num_actions, activation_fn=tf.nn.softmax)
            add_summaries = not reuse
            if add_summaries:
                tf.contrib.layers.summarize_activation(out)

        return out

class TFBrain(IBrain): 

    def __init__(self, num_actions, lr, stateShape):
        self.num_actions = num_actions
        self.lr = lr
        self.debugMode = True

        self.Actor = Actor(num_actions, lr, stateShape)
        self.Critic = Critic(num_actions, lr, stateShape)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        # THIS DOESNT WORK, NOT SURE WHY
        self.global_step = tf.train.get_global_step()

        self.stepCount = 0
        self.summary_writer = tf.summary.FileWriter('tf_train', self.sess.graph)

        
    def train(self, states, actions, Rs, calc_diff):
        self.stepCount += len(states)
        diff = None

        # We should split the 2 models into 2 classes to prevent this
        fixedUpRs = np.array(Rs).flatten()
        v_feed_dict = {self.Critic.state: states}

        v_calcs = np.array(self.sess.run(self.Critic.prediction_op, v_feed_dict)).flatten()

        if self.debugMode:
            print("v_calcs: ", v_calcs)

        actor_train_feed_dict = {self.Actor.state: states, self.Actor.action: (actions), self.Actor.R: (fixedUpRs), self.Actor.v_calc: v_calcs}
        _, _, actor_summaries = self.sess.run(
            [self.Actor.loss_op, self.Actor.train_op, self.Actor.summaries], 
            actor_train_feed_dict
            )

        critic_train_feed_dict = {self.Critic.state: states, self.Critic.R: fixedUpRs}
        _, _, critic_summaries = self.sess.run(
            [self.Critic.loss_op, self.Critic.train_op, self.Critic.summaries], 
            critic_train_feed_dict
            )

        # Write summaries
        self.summary_writer.add_summary(actor_summaries, self.stepCount)
        self.summary_writer.add_summary(critic_summaries, self.stepCount)
        self.summary_writer.flush()

        return
        
    # Called
    def state_value(self, state):
        feed_dict = {self.Critic.state: [state]}
        return self.sess.run(self.Critic.prediction_op, feed_dict)
    
    # Called by Agent
    def pi_probabilities(self, state):
        feed_dict = {self.Actor.state: [state]}
        probs_to_return = self.sess.run(self.Actor.prediction_op, feed_dict)
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

    def __del__(self):
        print("Closing tf session")
        self.sess.close()