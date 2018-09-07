from __future__ import print_function
import sys
import time

from agents.base_agent import BaseAgent # for python3
#from base_agent import BaseAgent # for python2

import random
import numpy as np
import tensorflow as tf

from agents.network.actor_network import ActorNetwork
from agents.network.critic_network import CriticNetwork
from utils.running_mean_std import RunningMeanStd
from experiment import write_summary
import utils.plot_utils

class DDPG_Network(object):
    def __init__(self, state_dim, state_min, state_max, action_dim, action_min, action_max, config, random_seed):
        
        # type of normalization: 'none', 'batch', 'layer', 'input_norm'
        self.norm_type = config.norm

        if self.norm_type is not 'none':
            self.input_norm = RunningMeanStd(state_dim)
        else:
            assert(self.norm_type == 'none')
            self.input_norm = None

        self.action_min = action_min
        self.action_max = action_max

        self.train_global_steps = 0
        self.eval_global_steps = 0
        self.train_ep_count = 0
        self.eval_ep_count = 0

        self.write_log = config.write_log
        self.write_plot = config.write_plot


        self.episode_ave_max_q = 0.0
        self.graph = tf.Graph()

        with self.graph.as_default():
            tf.set_random_seed(random_seed)
            self.sess = tf.Session() 

            actor_layer_dim = [config.actor_l1_dim, config.actor_l2_dim]
            critic_layer_dim = [config.critic_l1_dim, config.critic_l2_dim]

            self.actor_network = ActorNetwork(self.sess, self.input_norm, actor_layer_dim, state_dim, state_min, state_max, action_dim, action_min, action_max,
                                              config.actor_lr, config.tau, norm_type=self.norm_type)
            self.critic_network = CriticNetwork(self.sess, self.input_norm, critic_layer_dim, state_dim, state_min, state_max, action_dim, action_min, action_max,
                                                config.critic_lr, config.tau, norm_type=self.norm_type)

            self.sess.run(tf.global_variables_initializer())

            self.actor_network.update_target_network()
            self.critic_network.update_target_network()

    def take_action(self, state, is_train, is_start):

        chosen_action = self.actor_network.predict(np.expand_dims(state, 0), False)[0]

        if not is_train:
            self.eval_global_steps += 1

            if self.write_log:
                write_summary(self.writer, self.eval_global_steps, chosen_action[0], tag='eval/action_taken')

            # if self.write_plot:
            #
            #     if is_start:
            #         self.eval_ep_count += 1
            #
            #     if self.eval_global_steps % 1 == 0:
            #         func1 = self.critic_network.getQFunction(state)
            #
            #         utils.plot_utils.plotFunction("DDPG", [func1], state, chosen_action, self.action_min, self.action_max,
            #                            display_title='ep: ' + str(self.eval_ep_count) + ', steps: ' + str(self.eval_global_steps),
            #                            save_title='steps_' + str(self.eval_global_steps),
            #                            save_dir=self.writer.get_logdir(), ep_count=self.eval_ep_count, show=False)

        else:
            self.train_global_steps += 1
            ## MOVED OUTSIDE TO LOG BOTH EXPLORATION AND GREEDY ACTION
            # if self.write_plot:
            #
            #     if is_start:
            #         self.train_ep_count += 1
            #
            #     func1 = self.critic_network.getQFunction(state)
            #
            #     utils.plot_utils.plotFunction("DDPG", [func1], state, chosen_action, self.action_min, self.action_max,
            #                        display_title='ep: ' + str(self.train_ep_count) + ', steps: ' + str(self.train_global_steps),
            #                        save_title='steps_' + str(self.train_global_steps),
            #                        save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count, show=False)


        return chosen_action

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        # compute target
        target_q = self.critic_network.predict_target(next_state_batch, self.actor_network.predict_target(next_state_batch, True), True)

        batch_size = np.shape(state_batch)[0]
        reward_batch = np.reshape(reward_batch, (batch_size, 1))
        gamma_batch = np.reshape(gamma_batch, (batch_size, 1))
        target_q = np.reshape(target_q, (batch_size, 1))

        y_i = reward_batch + gamma_batch * target_q

        # Update the critic given the targets
        predicted_q_value, _ = self.critic_network.train(state_batch, action_batch, y_i)
        self.episode_ave_max_q += np.amax(predicted_q_value) 

        # Update the actor using the sampled gradient
        a_outs = self.actor_network.predict(state_batch, True)
        a_grads = self.critic_network.action_gradients(state_batch, a_outs, True)
        self.actor_network.train(state_batch, a_grads[0])

        # Update target networks
        self.actor_network.update_target_network()
        self.critic_network.update_target_network()

    def get_sum_maxQ(self): # Returns sum of max Q values
        return self.episode_ave_max_q

    def reset(self):
        self.episode_ave_max_q = 0.0


class DDPG(BaseAgent):
    def __init__(self, env, config, random_seed):
        super(DDPG, self).__init__(env, config)

        np.random.seed(random_seed)
        random.seed(random_seed)

        # Network
        self.network = DDPG_Network(self.state_dim, self.state_min, self.state_max, 
                                    self.action_dim, self.action_min, self.action_max,
                                    config, random_seed=random_seed)

        self.cum_steps = 0  # cumulative steps across episodes

    def start(self, state, is_train):
        return self.take_action(state, is_train, is_start=True)

    def step(self, state, is_train):
        return self.take_action(state, is_train, is_start=False)

    def take_action(self, state, is_train, is_start):

        # random action during warmup
        if self.cum_steps < self.warmup_steps:
            action = np.random.uniform(self.action_min, self.action_max)

        else:
            action = self.network.take_action(state, is_train, is_start)

            # Train
            if is_train:

                greedy_action = action
                # if using an external exploration policy
                if self.use_external_exploration:
                    action = self.exploration_policy.generate(greedy_action, self.cum_steps)
                
                # only increment during training, not evaluation
                self.cum_steps += 1

                # HACKY WAY
                if self.write_plot:

                    if is_start:
                        self.network.train_ep_count += 1

                    func1 = self.network.critic_network.getQFunction(state)

                    utils.plot_utils.plotFunction("DDPG", [func1], state, greedy_action, action, self.action_min,
                                                  self.action_max,
                                                  display_title='ep: ' + str(self.network.train_ep_count) + ', steps: ' + str(
                                                      self.network.train_global_steps),
                                                  save_title='steps_' + str(self.network.train_global_steps),
                                                  save_dir=self.writer.get_logdir(), ep_count=self.network.train_ep_count,
                                                  show=False)

            action = np.clip(action, self.action_min, self.action_max) 
        return action

    def update(self, state, next_state, reward, action, is_terminal, is_truncated): # hello from the other lab, saranghe!

        if not is_truncated:
            if not is_terminal:
                self.replay_buffer.add(state, action, reward, next_state, self.gamma)
            else:
                self.replay_buffer.add(state, action, reward, next_state, 0.0)

        if self.network.norm_type is not 'none':
            self.network.input_norm.update(np.array([state]))

        self.learn()
    
    def learn(self):

        if self.replay_buffer.get_size() > max(self.warmup_steps, self.batch_size):
            state, action, reward, next_state, gamma = self.replay_buffer.sample_batch(self.batch_size)
            self.network.update_network(state, action, next_state, reward, gamma)
        else:
            return

    # not implemented
    def get_Qfunction(self, state):
        raise NotImplementedError

    def reset(self):
        self.network.reset()
        if self.exploration_policy:
            self.exploration_policy.reset()



