import tensorflow as tf
import numpy as np
# import tf_cnnvis 

from tensorflow.python import debug as tf_debug
from Brain.IBrain import IBrain 
# tensorboard --logdir=tf_train  http://localhost:6006/

class BaseTFModel():
    def create_base_model(self, dropout, reuse, is_training):

        local_state = tf.reshape(self.state, shape=[-1, self.STATE_WIDTH, self.STATE_HEIGHT, 1])

        conv1 = tf.layers.conv2d(local_state, filters=8, kernel_size=(8, 8), strides=4, padding="VALID", use_bias=True, activation=tf.nn.relu, name="conv1")
        conv2 = tf.layers.conv2d(conv1, filters=16, kernel_size=(4, 4), strides=2, padding="VALID", use_bias=True, activation=tf.nn.relu, name="conv2")

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.contrib.layers.fully_connected(fc1, 256, activation_fn=tf.nn.sigmoid)

        # Apply Dropout (if is_training is False, dropout is not applied)
        # REMOVING DROPOUT TEMPORARILY
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        add_summaries = not reuse
        if add_summaries:
            tf.contrib.layers.summarize_activation(conv1)
            tf.contrib.layers.summarize_activation(conv2)
            tf.contrib.layers.summarize_activation(fc1)

            if self.debugMode:
                self.setupVisualizations(conv1, conv2, local_state)

        return fc1

    # https://github.com/tensorflow/tensorflow/issues/908
    def setupVisualizations(self, conv1, conv2, local_state):
 
        self.visualize_single_tensor(conv1, conv1.shape[1], conv1.shape[2], 4, 2, 'conv1')

        self.visualize_single_tensor(conv2, conv2.shape[1], conv2.shape[2], 4, 4, 'conv2')

        self.visualize_single_kernel("/conv1/kernel:0", 4, 2)

        self.visualize_single_kernel("/conv2/kernel:0", 8, 16)

        tf.summary.image("input_image", local_state)

    def visualize_single_kernel(self, layer_name, num_rows, num_kernels_per_row):

        conv_layer_weights = [v for v in tf.trainable_variables(tf.get_variable_scope().name) if v.name ==  tf.get_variable_scope().name + layer_name][0]

        kernel_height = conv_layer_weights.shape[0]
        kernel_width = conv_layer_weights.shape[1]
        kernel_colors = conv_layer_weights.shape[2]
        num_kernels = int(conv_layer_weights.shape[3].value)

        W1_c = tf.split(conv_layer_weights, num_kernels, 3)         # 36 x [5, 5, 1, 1]

        Weights_Flattened = []

        for kernel in W1_c: 
            Weights_Flattened += (tf.split(kernel, kernel_colors, 2))

        paddings = tf.constant([[1, 1], [1, 1,], [0, 0], [0, 0]]) # Pad along the 1st and 2nd dim

        for kernel_index in range(len(Weights_Flattened)):
            Weights_Flattened[kernel_index] = tf.pad(Weights_Flattened[kernel_index], paddings, "CONSTANT")

        arrayOfFilterRows = []
        for i in range(num_rows):
            W1_row = tf.concat(Weights_Flattened[num_kernels_per_row * i:(i + 1) * num_kernels_per_row ], 0)    # [30, 5, 1, 1]
            arrayOfFilterRows.append(W1_row)        
 
        W1_d = tf.concat(arrayOfFilterRows, 1) # [30, 30, 1, 1]
        W1_d = tf.reshape(W1_d, shape=[1, W1_d.shape[0], W1_d.shape[1], 1])

        tf.summary.image("Visualize_kernels_conv1_" + layer_name , W1_d)
    
    # https://stackoverflow.com/questions/33802336/visualizing-output-of-convolutional-layer-in-tensorflow
    def visualize_single_tensor(self, tensor, iy, ix, cy, cx, name):
        initTensor = tensor

        tensor = tf.slice(tensor,(0,0,0,0),(1,-1,-1,-1))
        tensor = tf.reshape(tensor,(iy,ix,cy*cx))
        tensor = tf.reshape(tensor,(iy,ix,cy,cx))
        tensor = tf.transpose(tensor,(2,0,3,1)) #cy,iy,cx,ix
        newtensor = tf.einsum('yxYX->YyXx',tensor)
        newtensor = tf.reshape(newtensor,(1,cy*iy,cx*ix,1))
        tf.summary.image("Visualize_output_" + name, newtensor)


class Critic(BaseTFModel):
    def __init__(self, num_actions, lr, stateShape, model_name):

        self.num_actions = num_actions
        self.lr = lr
        self.debugMode = True
        self.STATE_WIDTH = stateShape[0]
        self.STATE_HEIGHT = stateShape[1]
        self.model_name = model_name

        # Defining the input variables for training and evaluation.
        self.state = tf.placeholder(tf.float32, [None, 1, self.STATE_WIDTH, self.STATE_HEIGHT], "State")
        self.R = tf.placeholder(tf.float32, [None], "RewardR")
                
        self.dropout = 0.5
     
        with tf.variable_scope(self.model_name):
            # Create the train and test graphs
            logits_train = self.create_model(self.dropout, reuse = False, is_training = True)
            logits_test = self.create_model(self.dropout, reuse = True, is_training = False)

            # Display the image only 1ce

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
            tf.summary.scalar("Critic_nonzeroStates", tf.count_nonzero(self.state))
            tf.summary.histogram("{}/reward_targets".format(prefix), self.R)
            tf.summary.histogram("{}/values".format(prefix), logits_train)


        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries = [s for s in summary_ops if self.model_name in s.name]

        self.summaries = tf.summary.merge(summaries)

        return

    def create_model(self, dropout, reuse, is_training):
        with tf.variable_scope(self.model_name, reuse=reuse):
            fc1 = self.create_base_model(self.dropout, reuse, is_training)
            out = tf.contrib.layers.fully_connected(fc1, 1, activation_fn=tf.nn.sigmoid)

            add_summaries = not reuse
            if add_summaries:
                tf.contrib.layers.summarize_activation(out)

        return out

class Actor(BaseTFModel):
    def __init__(self, num_actions, lr, stateShape, model_name):

        self.num_actions = num_actions
        self.lr = lr
        self.debugMode = True
        self.STATE_WIDTH = stateShape[0]
        self.STATE_HEIGHT = stateShape[1]
        self.model_name = model_name

        # Defining the input variables for training and evaluation.
        self.state = tf.placeholder(tf.float32, [None, 1, self.STATE_WIDTH, self.STATE_HEIGHT], "State")
        self.action = tf.placeholder(tf.int32, [None], "Action")
        self.R = tf.placeholder(tf.float32, [None], "RewardR")
        self.v_calc = tf.placeholder(tf.float32, [None], "V_Calc")
                
        self.dropout = 0.1
   
        with tf.variable_scope(self.model_name):
            logits_train = self.create_model(self.dropout, reuse = False, is_training = True)
            logits_test = self.create_model(self.dropout, reuse = True, is_training = False)

            probs_fixed = logits_train + 1e-8
            test_probs_fixed = logits_test + 1e-8

            batch_size = tf.shape(self.state)[0]

            gather_indices = tf.range(batch_size) * tf.shape(probs_fixed)[1] + self.action
            picked_action_probs = tf.gather(tf.reshape(probs_fixed, [-1]), gather_indices)

            entropy = -tf.reduce_sum(probs_fixed * tf.log(probs_fixed), 1, name="actor_entropy")
            advantage = tf.reduce_sum(self.R - self.v_calc, name="advantage")
            self.loss_op = tf.reduce_sum((-(tf.log(picked_action_probs) * advantage + 0.01 * entropy)), name="actor_loss")


            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(self.loss_op,
                                          global_step=tf.train.get_global_step())

            self.prediction_op = test_probs_fixed


            tf.summary.scalar(self.loss_op.name, self.loss_op)
            tf.summary.scalar(advantage.name, advantage)
            tf.summary.scalar("Actor_nonzeroStates", tf.count_nonzero(self.state))
            tf.summary.histogram(entropy.op.name, entropy)

            # tf_cnnvis.activation_visualization(sess_graph_path = tf.get_default_graph(), value_feed_dict = {X : im}, 
            #                               layers=layers, path_logdir=os.path.join("Log","AlexNet"), 
            #                               path_outdir=os.path.join("Output","AlexNet"))

        
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries = [s for s in summary_ops if self.model_name in s.name]

        self.summaries = tf.summary.merge(summaries)

        return

    def create_model(self, dropout, reuse, is_training):        
        with tf.variable_scope(self.model_name, reuse=reuse):
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

        self.Actor_local = Actor(num_actions, lr, stateShape, "ActorModel_local")
        self.Critic_local = Critic(num_actions, lr, stateShape, "CriticModel_local")

        self.Actor_global = Actor(num_actions, lr, stateShape, "ActorModel_global")
        self.Critic_global= Critic(num_actions, lr, stateShape, "CriticModel_global")

        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        self.sess = tf.Session(config=config)
        # if self.debugMode:
        #     self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

        self.sess.run(init)

        # THIS DOESNT WORK, NOT SURE WHY
        self.global_step = tf.train.get_global_step()

        self.stepCount = 0
        self.summary_writer = tf.summary.FileWriter('tf_train', self.sess.graph)


    # Copies one set of variables to another.
    # Used to set worker network parameters to those of global network.
    def update_target_graph(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder
    
    def train(self, states, actions, Rs, calc_diff):
        self.stepCount += 1
        diff = None

        # We should split the 2 models into 2 classes to prevent this
        fixedUpRs = np.array(Rs).flatten()
        v_feed_dict = {self.Critic_local.state: states}

        v_calcs = np.array(self.sess.run(self.Critic_local.prediction_op, v_feed_dict)).flatten()

        if self.debugMode:
            print("v_calcs: ", v_calcs)

        actor_train_feed_dict = {
            self.Actor_local.state: states, 
            self.Actor_local.action: (actions), 
            self.Actor_local.R: (fixedUpRs), 
            self.Actor_local.v_calc: v_calcs,
            }

        _, _, actor_summaries = self.sess.run(
            [self.Actor_local.loss_op, self.Actor_local.train_op, self.Actor_local.summaries],
            actor_train_feed_dict
            )

        critic_train_feed_dict = {
            self.Critic_local.state: states, 
            self.Critic_local.R: fixedUpRs,
            }

        _, _, critic_summaries = self.sess.run(
            [self.Critic_local.loss_op, self.Critic_local.train_op, self.Critic_local.summaries],
            critic_train_feed_dict
            )

        # if self.stepCount % 100 == 0:
        #     update_global_graphs = self.update_target_graph(self.Actor_local.model_name, self.Actor_global.model_name)
        #     update_global_graphs += self.update_target_graph(self.Critic_local.model_name, self.Critic_global.model_name)

        #     self.sess.run(update_global_graphs)

        # Write summaries
        self.summary_writer.add_summary(actor_summaries,  (self.stepCount))
        self.summary_writer.add_summary(critic_summaries, (self.stepCount))
        self.summary_writer.flush()

        return
        
    # Called
    def state_value(self, state):
        feed_dict = {self.Critic_local.state: [state]}
        return self.sess.run(self.Critic_local.prediction_op, feed_dict)
    
    # Called by Agent
    def pi_probabilities(self, state):
        feed_dict = {self.Actor_local.state: [state]}
        probs_to_return = self.sess.run(self.Actor_local.prediction_op, feed_dict)
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