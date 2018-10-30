class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1, 160, 160, 3])
            
            self.conv1 = tf.layers.conv2d(self.imageIn, filters=8, kernel_size=(8, 8), strides=4, padding="VALID", use_bias=True, activation=tf.nn.relu, name="conv1")
            self.conv2 = tf.layers.conv2d(self.conv1, filters=16, kernel_size=(4, 4), strides=2, padding="VALID", use_bias=True, activation=tf.nn.relu, name="conv2")
            self.conv3 = tf.layers.conv2d(self.conv2, filters=16, kernel_size=(4, 4), strides=2, padding="VALID", use_bias=True, activation=tf.nn.relu, name="conv3")

            hidden = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.conv3), 256, activation_fn=tf.nn.sigmoid)

            #Output layers for policy and value estimations
            self.policy = tf.contrib.layers.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax)

            self.value = tf.contrib.layers.fully_connected(hidden, 1, activation_fn=tf.nn.leaky_relu)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))