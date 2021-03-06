import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
import environments.environments


class AE_Actor_Network(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(AE_Actor_Network, self).__init__(sess, config, config.actor_lr)

        self.rng = np.random.RandomState(config.random_seed)

        self.actor_layer1_dim = config.l1_dim
        self.actor_layer2_dim = config.l2_dim

        self.input_norm = input_norm

        # ae specific params
        self.rho = config.rho
        self.num_samples = config.num_samples
        self.num_modal = config.num_modal
        self.actor_output_dim = self.num_modal * (1 + 2 * self.action_dim)

        self.sigma_scale = 1.0  # config.sigma_scale

        self.use_uniform_sampling = False
        if config.use_uniform_sampling == "True":
            self.use_uniform_sampling = True
            self.uniform_sampling_ratio = 0.2  # config.uniform_sampling_ratio

        self.use_better_q_gd = False
        if config.use_better_q_gd == "True":
            self.use_better_q_gd = True
            self.better_q_gd_alpha = 1e-2  # config.better_q_gd_alpha
            self.better_q_gd_max_steps = 10  # config.better_q_gd_max_steps
            self.better_q_gd_stop = 1e-3  # config.better_q_gd_stop

        # Removed from config
        # "better_q_gd_alpha": [1e-2],
        # "better_q_gd_max_steps": [10],
        # "better_q_gd_stop": [1e-3]

        # currently not used
        self.use_policy_gd = False
        # if config.use_policy_gd == "True":
        #     self.use_policy_gd = True
        #     self.policy_gd_alpha = config.policy_gd_alpha
        #     self.policy_gd_max_steps = config.policy_gd_max_steps
        #     self.policy_gd_stop = config.policy_gd_stop

        # Removed from config
        # "use_policy_gd": ["False"],
        # "policy_gd_alpha": [1e-1],
        # "policy_gd_max_steps": [50],
        # "policy_gd_stop": [1e-3],


        # original network
        self.inputs, self.phase, self.action, self.action_prediction_mean, self.action_prediction_sigma, self.action_prediction_alpha = self.build_network(
            scope_name='ae_actor')
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ae_actor')

        # Target network
        self.target_inputs, self.target_phase, self.target_action, self.target_action_prediction_mean, self.target_action_prediction_sigma, self.target_action_prediction_alpha = self.build_network(
            scope_name='target_ae_actor')
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_ae_actor')

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

        self.actions = tf.placeholder(tf.float32, [None, self.action_dim])

        # Optimization Op
        with tf.control_dependencies(self.batchnorm_ops):
            # Actor update
            self.actor_loss = self.get_lossfunc(self.action_prediction_alpha, self.action_prediction_sigma,
                                                self.action_prediction_mean, self.actions)
            self.actor_optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.actor_loss)

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

            action_prediction_mean, action_prediction_sigma, action_prediction_alpha = self.network(inputs, phase)

        return inputs, phase, action, action_prediction_mean, action_prediction_sigma, action_prediction_alpha

    def network(self, inputs, phase):
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
            tf.sqrt(1.0 / (2 * np.pi * tf.square(sigma))) * tf.exp(-tf.square(stacked_y - mu) / (2 * tf.square(sigma))),
            axis=2)

    def get_policyfunc(self):

        alpha = tf.placeholder(tf.float32, shape=(None, self.num_modal, 1), name='temp_alpha')
        mean = tf.placeholder(tf.float32, shape=(None, self.num_modal, self.action_dim), name='temp_mean')
        sigma = tf.placeholder(tf.float32, shape=(None, self.num_modal, self.action_dim), name='temp_sigma')
        action = tf.placeholder(tf.float32, shape=(None, self.action_dim), name='temp_action')

        result = self.tf_normal(action, mean, sigma)
        result = tf.multiply(result, tf.squeeze(alpha, axis=2))

        result = tf.reduce_sum(result, 1, keepdims=True)
        return alpha, mean, sigma, action, result

    def get_lossfunc(self, alpha, sigma, mean, action):
        # alpha: batch x num_modal x 1
        # sigma: batch x num_modal x action_dim
        # mean: batch x num_modal x action_dim
        # action: batch x action_dim
        result = self.tf_normal(action, mean, sigma)
        result = tf.multiply(result, tf.squeeze(alpha, axis=2))

        result = tf.reduce_sum(result, 1, keepdims=True)
        result = -tf.log(tf.clip_by_value(result, 1e-30, 1e30))

        return tf.reduce_mean(result)

    def policy_gradient_ascent(self, alpha, mean, sigma, action_init):

        action = np.copy(action_init)

        ascent_count = 0
        update_flag = np.ones([alpha.shape[0], self.action_dim])  # batch_size * action_dim

        while np.any(update_flag > 0) and ascent_count < self.policy_gd_max_steps:
            action_old = np.copy(action)

            # print(np.shape(self.policy_action_gradients(alpha, mean, sigma, action)))
            # exit()
            gradients = self.policy_action_gradients(alpha, mean, sigma, action)[0]
            action += update_flag * self.policy_gd_alpha * gradients
            action = np.clip(action, self.action_min, self.action_max)

            # stop if action diff. is small
            stop_idx = [idx for idx in range(len(action)) if
                        np.mean(np.abs(action_old[idx] - action[idx]) / self.action_max) <= self.policy_gd_stop]
            update_flag[stop_idx] = 0
            # print(update_flag)

            ascent_count += 1

        # print('ascent count:', ascent_count)
        return action

    def policy_action_gradients(self, alpha, mean, sigma, action):

        # print(np.shape(alpha), type(self.temp_alpha))
        # print(np.shape(mean), type(self.temp_mean))
        # print(np.shape(sigma), type(self.temp_sigma))
        # print(np.shape(action), type(self.temp_action))

        return self.sess.run(self.policy_action_grads, feed_dict={
            self.temp_alpha: alpha,
            self.temp_mean: mean,
            self.temp_sigma: sigma,
            self.temp_action: action
        })

    def train_actor(self, *args):
        # args [inputs, actions, phase]
        return self.sess.run(self.actor_optimize, feed_dict={
            self.inputs: args[0],
            self.actions: args[1],
            self.phase: True
        })

    def predict_true_q(self, *args):
        # args  (inputs, action, phase)
        inputs = args[0]
        action = args[1]
        phase = args[2]
        env_name = args[3]

        return [getattr(environments.environments, env_name).reward_func(a[0]) for a in action]

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
        self.setModalStats(alpha[0], mean[0], sigma[0])
        max_idx = np.argmax(np.squeeze(alpha, axis=2), axis=1)

        best_mean = [m[idx] for idx, m in zip(max_idx, mean)]

        old_best_mean = best_mean
        if self.use_policy_gd:
            best_mean = self.policy_gradient_ascent(alpha, mean, sigma, best_mean)

        return old_best_mean, best_mean

    # Should return n actions
    def sample_action(self, inputs, phase, is_single_sample):
        # args [inputs]

        # batchsize x action_dim
        alpha, mean, sigma = self.sess.run(
            [self.action_prediction_alpha, self.action_prediction_mean, self.action_prediction_sigma], feed_dict={
                self.inputs: inputs,
                self.phase: phase
            })

        alpha = np.squeeze(alpha, axis=2)

        self.setModalStats(alpha[0], mean[0], sigma[0])

        # selected_idx = np.random.choice(self.num_modal, self.num_samples, p=alpha[0])

        if is_single_sample:
            num_samples = 1
        else:
            num_samples = self.num_samples

        modal_idx_list = [self.rng.choice(self.num_modal, num_samples, p=prob) for prob in alpha]
        sampled_actions = [np.clip(self.rng.normal(m[idx], s[idx]), self.action_min, self.action_max) for idx, m, s
                           in zip(modal_idx_list, mean, sigma)]

        # uniform sampling TODO: Optimize this
        if self.use_uniform_sampling and not is_single_sample:
            for j in range(len(sampled_actions)):
                for i in range(int(self.num_samples * self.uniform_sampling_ratio)):
                    sampled_actions[j][i] = self.rng.uniform(self.action_min, self.action_max)

        return sampled_actions

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    def getPolicyFunction(self, alpha, mean, sigma):

        # alpha = np.squeeze(alpha, axis=1)
        mean = np.squeeze(mean, axis=1)
        sigma = np.squeeze(sigma, axis=1)

        return lambda action: np.sum(alpha * np.multiply(np.sqrt(1.0 / (2 * np.pi * np.square(sigma))), np.exp(-np.square(action - mean) / (2.0 * np.square(sigma)))))

    def setModalStats(self, alpha, mean, sigma):
        self.saved_alpha = alpha
        self.saved_mean = mean
        self.saved_sigma = sigma

    def getModalStats(self):
        return self.saved_alpha, self.saved_mean, self.saved_sigma
