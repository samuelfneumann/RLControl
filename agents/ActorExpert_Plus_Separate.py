from __future__ import print_function
import random
import numpy as np
import tensorflow as tf

from agents.base_agent import BaseAgent
from agents.network.base_network_manager import BaseNetwork_Manager

from agents.network import ae_plus_actor_network
from agents.network import ae_plus_expert_network
from experiment import write_summary
import utils.plot_utils


class ActorExpert_Plus_Separate_Network_Manager(BaseNetwork_Manager):
    def __init__(self, config):
        super(ActorExpert_Plus_Separate_Network_Manager, self).__init__(config)

        self.rng = np.random.RandomState(config.random_seed)

        # Cross Entropy Method Params
        self.rho = config.rho
        self.num_samples = config.num_samples

        with self.graph.as_default():
            tf.set_random_seed(config.random_seed)
            self.sess = tf.Session()
            # self.hydra_network = ae_plus_network.ActorExpert_Plus_Network(self.sess, self.input_norm, config)
            self.actor_network = ae_plus_actor_network.ActorExpert_Plus_Actor_Network(self.sess, self.input_norm, config)
            self.expert_network = ae_plus_expert_network.ActorExpert_Plus_Expert_Network(self.sess, self.input_norm, config)
            self.sess.run(tf.global_variables_initializer())

            self.actor_network.init_target_network()
            self.expert_network.init_target_network()

    def take_action(self, state, is_train, is_start):

        if is_train:
            if is_start:
                self.train_ep_count += 1

            if self.use_external_exploration:
                _, greedy_action = self.actor_network.predict_action(np.expand_dims(state, 0), False)
                chosen_action = self.exploration_policy.generate(greedy_action[0], self.train_global_steps)

            else:
                # single state so first idx
                chosen_action = self.actor_network.sample_action(np.expand_dims(state, 0), False, is_single_sample=True)[0][0]

            self.train_global_steps += 1

            if self.write_log:
                write_summary(self.writer, self.train_global_steps, chosen_action[0], tag='train/action_taken')

            if self.write_plot:
                alpha, mean, sigma = self.actor_network.getModalStats()
                func1 = self.expert_network.getQFunction(state)
                func2 = self.actor_network.getPolicyFunction(alpha, mean, sigma)

                old_greedy_action, greedy_action = self.actor_network.predict_action(np.expand_dims(state, 0), False)

                if self.actor_network.use_better_q_gd:
                    greedy_action = self.expert_network.q_gradient_ascent(np.expand_dims(state, 0), greedy_action, True, is_better_q_gd=True)

                old_greedy_action = old_greedy_action[0]
                greedy_action = greedy_action[0]

                utils.plot_utils.plotFunction("ActorExpert", [func1, func2], state, [greedy_action, old_greedy_action, mean], chosen_action,
                                              self.action_min, self.action_max,
                                              display_title='Actor-Expert+ , steps: ' + str(
                                                  self.train_global_steps),
                                              save_title='steps_' + str(self.train_global_steps),
                                              save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count,
                                              show=False)
        else:

            old_greedy_action, greedy_action = self.actor_network.predict_action(np.expand_dims(state, 0), False)

            if self.actor_network.use_better_q_gd:
                greedy_action = self.expert_network.q_gradient_ascent(np.expand_dims(state, 0), greedy_action, True,
                                                                     is_better_q_gd=True)

            old_greedy_action = old_greedy_action[0]
            greedy_action = greedy_action[0]

            if is_start:
                self.eval_ep_count += 1

            chosen_action = greedy_action
            self.eval_global_steps += 1

            if self.write_log:
                write_summary(self.writer, self.eval_global_steps, chosen_action[0], tag='eval/action_taken')

        return chosen_action

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        batch_size = np.shape(state_batch)[0]

        # Expert Update
        # Perhaps do GA on the Q function
        _, next_action_batch_init_target = self.actor_network.predict_action(next_state_batch, True)

        if self.actor_network.use_better_q_gd:
            next_action_batch_final_target = self.expert_network.q_gradient_ascent(next_state_batch, next_action_batch_init_target, True, is_better_q_gd=True)
        else:
            next_action_batch_final_target = next_action_batch_init_target

        # batchsize * n
        target_q = self.expert_network.predict_q_target(next_state_batch, next_action_batch_final_target, True)

        reward_batch = np.reshape(reward_batch, (batch_size, 1))
        gamma_batch = np.reshape(gamma_batch, (batch_size, 1))

        # compute target : y_i = r_{i+1} + \gamma * max Q'(s_{i+1}, a')
        y_i = reward_batch + gamma_batch * target_q

        predicted_q_val, _ = self.expert_network.train_expert(state_batch, action_batch, y_i)

        # Actor Update

        # for each transition, n sampled actions
        # shape: (batchsize , n actions, action_dim)
        action_batch_init = self.actor_network.sample_action(state_batch, True, is_single_sample=False)
        # reshape (batchsize * n , action_dim)
        action_batch_init_reshaped =np.reshape(action_batch_init, (batch_size * self.num_samples, self.action_dim))

        # Currently using Current state batch instead of next state batch
        # (batchsize * n action values)
        # restack states (batchsize * n, 1)
        stacked_state_batch = np.repeat(state_batch, self.num_samples, axis=0)

        # Gradient Ascent
        action_batch_final_reshaped = self.expert_network.q_gradient_ascent(stacked_state_batch, action_batch_init_reshaped, True)  # do ascent on original network

        q_val = self.expert_network.predict_q(stacked_state_batch, action_batch_final_reshaped, True)
        q_val = np.reshape(q_val, (batch_size, self.num_samples))

        action_batch_final = np.reshape(action_batch_final_reshaped, (batch_size, self.num_samples, self.action_dim))

        # Find threshold : top (1-rho) percentile
        selected_idxs = list(map(lambda x: x.argsort()[::-1][:int(self.num_samples * self.rho)], q_val))

        action_list = [actions[idxs] for actions, idxs in zip(action_batch_final, selected_idxs)]

        # restack states (batchsize * top_idx_num, 1)
        stacked_state_batch = np.repeat(state_batch, int(self.num_samples * self.rho), axis=0)

        action_list = np.reshape(action_list, (batch_size * int(self.num_samples * self.rho), self.action_dim))
        self.actor_network.train_actor(stacked_state_batch, action_list)

        # Update target networks
        self.actor_network.update_target_network()
        self.expert_network.update_target_network()


class ActorExpert_Plus_Separate(BaseAgent):
    def __init__(self, config):

        network_manager = ActorExpert_Plus_Separate_Network_Manager(config)
        super(ActorExpert_Plus_Separate, self).__init__(config, network_manager)



