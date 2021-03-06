import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
import environments.environments


class AC_Actor_Network(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(AC_Actor_Network, self).__init__(sess, config, config.actor_lr)

        self.rng = np.random.RandomState(config.random_seed)

        self.actor_layer1_dim = config.l1_dim
        self.actor_layer2_dim = config.l2_dim

        self.input_norm = input_norm

        # ac specific params
        self.actor_update = config.actor_update

        self.num_modal = config.num_modal
        self.num_samples = config.num_samples
        self.actor_output_dim = self.num_modal * (1 + 2 * self.action_dim)
        self.sigma_scale = 2.0  # config.sigma_scale

        self.equal_modal_selection = False
        if config.equal_modal_selection == "True":
            self.equal_modal_selection = True

        # original network
        self.inputs, self.phase, self.action, self.action_prediction_mean, self.action_prediction_sigma, self.action_prediction_alpha = self.build_network(
            scope_name='ac_actor')
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ac_actor')

        # Target network
        self.target_inputs, self.target_phase, self.target_action, self.target_action_prediction_mean, self.target_action_prediction_sigma, self.target_action_prediction_alpha = self.build_network(
            scope_name='target_ac_actor')
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_ac_actor')

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = [
            tf.assign_add(self.target_net_params[idx], self.tau * (self.net_params[idx] - self.target_net_params[idx]))
            for idx in range(len(self.target_net_params))]

        # Op for init. target network with identical parameter as the original network
        self.init_target_net_params = [tf.assign(self.target_net_params[idx], self.net_params[idx]) for idx in
                                       range(len(self.target_net_params))]

        # TODO: Currently doesn't support batchnorm
        if self.norm_type == 'batch':
            raise NotImplementedError

        else:
            assert (self.norm_type == 'none' or self.norm_type == 'layer' or self.norm_type == 'input_norm')
            self.batchnorm_ops = [tf.no_op()]
            self.update_target_batchnorm_params = tf.no_op()

        self.q_val = tf.placeholder(tf.float32, [None, 1])
        self.actions = tf.placeholder(tf.float32, [None, self.action_dim])

        # Optimization Op
        with tf.control_dependencies(self.batchnorm_ops):

            # Actor update
            # Loglikelihood
            self.actor_loss_ll = self.get_lossfunc_ll(self.action_prediction_alpha, self.action_prediction_sigma,
                                                self.action_prediction_mean, self.actions, self.q_val)
            self.actor_optimize_ll = tf.train.AdamOptimizer(self.learning_rate).minimize(self.actor_loss_ll)

            # CEM
            self.actor_loss_cem = self.get_lossfunc_cem(self.action_prediction_alpha, self.action_prediction_sigma,
                                                self.action_prediction_mean, self.actions)
            self.actor_optimize_cem = tf.train.AdamOptimizer(self.learning_rate).minimize(self.actor_loss_cem)

        # # Get the gradient of the policy w.r.t. the action
        self.temp_alpha, self.temp_mean, self.temp_sigma, self.temp_action, self.policy_func = self.get_policyfunc()
        self.policy_action_grads = tf.gradients(self.policy_func, self.temp_action)

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):
            inputs = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="network_input_state")
            phase = tf.placeholder(tf.bool, name="network_input_phase")
            action = tf.placeholder(tf.float32, shape=(None, self.action_dim), name="network_input_action")

            # normalize inputs
            if self.norm_type != 'none':
                inputs = tf.clip_by_value(self.input_norm.normalize(inputs), self.state_min, self.state_max)

            action_prediction_mean, action_prediction_sigma, action_prediction_alpha = self.network(
                inputs, action, phase)

        return inputs, phase, action, action_prediction_mean, action_prediction_sigma, action_prediction_alpha

    def network(self, inputs, action, phase):
        # shared net
        action_net = tf.contrib.layers.fully_connected(inputs, self.actor_layer1_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        action_net = self.apply_norm(action_net, activation_fn=tf.nn.relu, phase=phase, layer_num=1)

        # action branch
        action_net = tf.contrib.layers.fully_connected(action_net, self.actor_layer2_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       # tf.truncated_normal_initializer(),
                                                       weights_regularizer=None,
                                                       # tf.contrib.layers.l2_regularizer(0.001),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        action_net = self.apply_norm(action_net, activation_fn=tf.nn.relu, phase=phase, layer_num=2)

        action_prediction_mean = tf.contrib.layers.fully_connected(action_net, self.num_modal * self.action_dim,
                                                                   activation_fn=tf.tanh,
                                                                   weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                                       factor=1.0, mode="FAN_IN", uniform=True),
                                                                   # weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                                   weights_regularizer=None,
                                                                   # tf.contrib.layers.l2_regularizer(0.001),
                                                                   biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                                       factor=1.0, mode="FAN_IN", uniform=True))
        # biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        action_prediction_sigma = tf.contrib.layers.fully_connected(action_net, self.num_modal * self.action_dim,
                                                                    activation_fn=tf.tanh,
                                                                    weights_initializer=tf.random_uniform_initializer(0,
                                                                                                                      3e-3),
                                                                    weights_regularizer=None,
                                                                    # tf.contrib.layers.l2_regularizer(0.001),
                                                                    biases_initializer=tf.random_uniform_initializer(
                                                                        -3e-3, 3e-3))

        action_prediction_alpha = tf.contrib.layers.fully_connected(action_net, self.num_modal, activation_fn=tf.tanh,
                                                                    weights_initializer=tf.random_uniform_initializer(
                                                                        -3e-3, 3e-3),
                                                                    weights_regularizer=None,
                                                                    # tf.contrib.layers.l2_regularizer(0.001),
                                                                    biases_initializer=tf.random_uniform_initializer(
                                                                        -3e-3, 3e-3))

        # reshape output
        action_prediction_mean = tf.reshape(action_prediction_mean, [-1, self.num_modal, self.action_dim])
        action_prediction_sigma = tf.reshape(action_prediction_sigma, [-1, self.num_modal, self.action_dim])
        action_prediction_alpha = tf.reshape(action_prediction_alpha, [-1, self.num_modal, 1])

        # scale mean to env. action domain
        action_prediction_mean = tf.multiply(action_prediction_mean, self.action_max)

        # exp. sigma
        action_prediction_sigma = tf.exp(tf.scalar_mul(self.sigma_scale, action_prediction_sigma))

        # mean: [None, num_modal, action_dim]  : [None, 1]
        # sigma: [None, num_modal, action_dim] : [None, 1]
        # alpha: [None, num_modal, 1]              : [None, 1]

        # compute softmax prob. of alpha
        max_alpha = tf.reduce_max(action_prediction_alpha, axis=1, keepdims=True)
        action_prediction_alpha = tf.subtract(action_prediction_alpha, max_alpha)
        action_prediction_alpha = tf.exp(action_prediction_alpha)

        normalize_alpha = tf.reciprocal(tf.reduce_sum(action_prediction_alpha, axis=1, keepdims=True))
        action_prediction_alpha = tf.multiply(normalize_alpha, action_prediction_alpha)

        return action_prediction_mean, action_prediction_sigma, action_prediction_alpha

    def tf_normal(self, y, mu, sigma):

        # y: batch x action_dim
        # mu: batch x num_modal x action_dim
        # sigma: batch x num_modal x action_dim

        # stacked y: batch x num_modal x action_dim
        stacked_y = tf.expand_dims(y, 1)
        stacked_y = tf.tile(stacked_y, [1, self.num_modal, 1])

        return tf.reduce_prod(
            tf.sqrt(1.0 / (2 * np.pi * tf.square(sigma))) * tf.exp(-tf.square(stacked_y - mu) / (2 * tf.square(sigma))), axis=2)

    def get_policyfunc(self):

        alpha = tf.placeholder(tf.float32, shape=(None, self.num_modal, 1), name='temp_alpha')
        mean = tf.placeholder(tf.float32, shape=(None, self.num_modal, self.action_dim), name='temp_mean')
        sigma = tf.placeholder(tf.float32, shape=(None, self.num_modal, self.action_dim), name='temp_sigma')
        action = tf.placeholder(tf.float32, shape=(None, self.action_dim), name='temp_action')

        result = self.tf_normal(action, mean, sigma)
        result = tf.multiply(result, tf.squeeze(alpha, axis=2))
        result = tf.reduce_sum(result, 1, keepdims=True)

        return alpha, mean, sigma, action, result

    def get_lossfunc_ll(self, alpha, sigma, mu, y, q_val):
        # alpha: batch x num_modal x 1
        # sigma: batch x num_modal x action_dim
        # mu: batch x num_modal x action_dim
        # y: batch x action_dim
        # q_val: batch x 1

        result = self.tf_normal(y, mu, sigma)

        # Modified to do equal weighting
        if self.equal_modal_selection:
            result = tf.scalar_mul(1.0 / self.num_modal, result)
        else:
            result = tf.multiply(result, tf.squeeze(alpha, axis=2))

        result = tf.reduce_sum(result, 1, keepdims=True)
        result = -tf.log(tf.clip_by_value(result, 1e-30, 1e30))
        result = tf.multiply(result, q_val)

        return tf.reduce_mean(result)

    def get_lossfunc_cem(self, alpha, sigma, mean, action):
        # alpha: batch x num_modal x 1
        # sigma: batch x num_modal x action_dim
        # mean: batch x num_modal x action_dim
        # action: batch x action_dim
        result = self.tf_normal(action, mean, sigma)

        # Modified to do equal weighting
        if self.equal_modal_selection:
            result = tf.scalar_mul(1.0 / self.num_modal, result)
        else:
            result = tf.multiply(result, tf.squeeze(alpha, axis=2))

        result = tf.reduce_sum(result, 1, keepdims=True)
        result = -tf.log(tf.clip_by_value(result, 1e-30, 1e30))

        return tf.reduce_mean(result)

    def policy_action_gradients(self, alpha, mean, sigma, action):

        return self.sess.run(self.policy_action_grads, feed_dict={
            self.temp_alpha: alpha,
            self.temp_mean: mean,
            self.temp_sigma: sigma,
            self.temp_action: action
        })

    def train_actor_ll(self, *args):
        # args [inputs, actions, phase]
        return self.sess.run(self.actor_optimize_ll, feed_dict={
            self.inputs: args[0],
            self.actions: args[1],
            self.q_val: args[2],
            self.phase: True
        })

    def train_actor_cem(self, *args):
        # args [inputs, actions, phase]
        return self.sess.run(self.actor_optimize_cem, feed_dict={
            self.inputs: args[0],
            self.actions: args[1],
            self.phase: True
        })

    # return sampled actions
    def sample_action(self, inputs, phase, is_single_sample):

        # batchsize x action_dim
        alpha, mean, sigma = self.sess.run(
            [self.action_prediction_alpha, self.action_prediction_mean, self.action_prediction_sigma], feed_dict={
                self.inputs: inputs,
                self.phase: phase
            })

        alpha = np.squeeze(alpha, axis=2)

        # self.setModalStats(alpha[0], mean[0], sigma[0])

        if is_single_sample:
            num_samples = None
        else:
            num_samples = self.num_samples

        if self.equal_modal_selection:
            modal_idx_list = [self.rng.choice(self.num_modal, size=num_samples) for _ in alpha]
        else:
            modal_idx_list = [self.rng.choice(self.num_modal, size=num_samples, p=prob) for prob in alpha]

        sampled_actions = [np.clip(self.rng.normal(m[idx], s[idx]), self.action_min, self.action_max) for idx, m, s
                          in zip(modal_idx_list, mean, sigma)]

        return sampled_actions

    # return uniformly sampled actions (batchsize x num_samples x action_dim)
    def sample_uniform_action(self, batchsize):
        return self.rng.uniform(self.action_min, self.action_max, size=(batchsize, self.num_samples, self.action_dim))

    def predict_action(self, *args):
        inputs = args[0]
        phase = args[1]

        # alpha: batchsize x num_modal x 1
        # mean: batchsize x num_modal x action_dim
        alpha, mean, sigma = self.sess.run(
            [self.action_prediction_alpha, self.action_prediction_mean, self.action_prediction_sigma], feed_dict={
                self.inputs: inputs,
                self.phase: phase
            })

        alpha = np.squeeze(alpha, axis=2)

        self.setModalStats(alpha[0], mean[0], sigma[0])

        if self.equal_modal_selection:
            max_idx = self.rng.randint(0, self.num_modal, size=len(mean))
        else:
            max_idx = np.argmax(alpha, axis=1)

        best_mean = [m[idx] for idx, m in zip(max_idx, mean)]

        return best_mean

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    def getPolicyFunction(self, alpha, mean, sigma):

        # alpha = np.squeeze(alpha, axis=1)
        mean = np.squeeze(mean, axis=1)
        sigma = np.squeeze(sigma, axis=1)

        if self.equal_modal_selection:
            return lambda action: np.sum((np.ones(self.num_modal) * (1.0 / self.num_modal)) * np.multiply(
                np.sqrt(1.0 / (2 * np.pi * np.square(sigma))),
                np.exp(-np.square(action - mean) / (2.0 * np.square(sigma)))))
        else:
            return lambda action: np.sum(alpha * np.multiply(
                np.sqrt(1.0 / (2 * np.pi * np.square(sigma))),
                np.exp(-np.square(action - mean) / (2.0 * np.square(sigma)))))

    def setModalStats(self, alpha, mean, sigma):
        self.saved_alpha = alpha
        self.saved_mean = mean
        self.saved_sigma = sigma

    def getModalStats(self):
        return self.saved_alpha, self.saved_mean, self.saved_sigma
