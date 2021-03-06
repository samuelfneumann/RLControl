from __future__ import print_function
import random
import numpy as np
import tensorflow as tf

from agents.base_agent import BaseAgent
from agents.network.base_network_manager import BaseNetwork_Manager
from agents.network import sac_network
from experiment import write_summary
import utils.plot_utils
# from spinup.utils.logx import EpochLogger
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


class SoftActorCritic_Network_Manager(BaseNetwork_Manager):
    def __init__(self, config):
        super(SoftActorCritic_Network_Manager, self).__init__(config)

        # self.logger = EpochLogger()
        self.rng = np.random.RandomState(config.random_seed)

        self.sample_for_eval = False
        if config.sample_for_eval == "True":
            self.sample_for_eval = True

        self.use_true_q = False
        if config.use_true_q == "True":
            self.use_true_q = True

        with self.graph.as_default():
            tf.set_random_seed(config.random_seed)
            self.sess = tf.Session()

            self.network = sac_network.SoftActorCriticNetwork(self.sess, self.input_norm, config)
            self.sess.run(tf.global_variables_initializer())

            if self.use_true_q:
                # load learned model
                ckpt_name = './Bimodal1DEnv_trueQ_ckpt/{}_trueQ_learned'.format(config.env_name)
                raw_vars_list = tf.train.list_variables(ckpt_name)

                vars_list = []
                for item in raw_vars_list:
                    if 'main/qf' in item[0]:
                        vars_list.append(item[0])

                variables_to_restore = self.network.get_variables_to_restore(tf.global_variables(), vars_list)
                restorer = tf.train.Saver(variables_to_restore)
                restorer.restore(self.sess, ckpt_name)

            self.network.init_target_network()



    def take_action(self, state, is_train, is_start):

        # Train
        if is_train:
            if is_start:
                self.train_ep_count += 1
            self.train_global_steps += 1

            if self.use_external_exploration:
                greedy_action = self.network.predict_action(np.expand_dims(state, 0))
                chosen_action = self.exploration_policy.generate(greedy_action[0], self.train_global_steps)

            else:
                # Get action from network
                chosen_action = self.network.sample_action(np.expand_dims(state, 0))[0]
                # print('train', chosen_action)

            if self.write_log:
                raise NotImplementedError

            if self.write_plot:

                if self.use_true_q:
                    # Loaded almost True Q
                    q_func = self.network.getQFunction(state)
                    # q_func = self.network.getTrueQFunction(state)
                    # raise NotImplementedError
                else:
                    q_func = self.network.getQFunction(state)
                pi_func = self.network.getPolicyFunction(state)
                greedy_action = self.network.predict_action(np.expand_dims(state, 0))[0]

                utils.plot_utils.plotFunction("SoftActorCritic", [q_func, pi_func], state,
                                              greedy_action, chosen_action,
                                              self.action_min, self.action_max,
                                              display_title='SoftActorCritic, steps: ' + str(self.train_global_steps),
                                              save_title='steps_' + str(self.train_global_steps),
                                              save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count,
                                              show=False)
        # Eval
        else:
            if self.sample_for_eval:
                # sample action
                chosen_action = self.network.sample_action(np.expand_dims(state, 0))[0]

            else:
                # greedy action (mean)
                chosen_action = self.network.predict_action(np.expand_dims(state, 0))[0]

            if is_start:
                self.eval_ep_count += 1
            self.eval_global_steps += 1

            if self.write_log:
                write_summary(self.writer, self.eval_global_steps, chosen_action[0], tag='eval/action_taken')

        return chosen_action

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        if self.use_true_q:
            # outs = self.network.update_network_true_q(state_batch, action_batch, next_state_batch, reward_batch, gamma_batch)
            outs = self.network.update_network(state_batch, action_batch, next_state_batch, reward_batch, gamma_batch)
        else:
            # Policy Update, Qf and Vf Update
            outs = self.network.update_network(state_batch, action_batch, next_state_batch, reward_batch, gamma_batch)

            # self.logger.store(LossPi=outs[0], LossQ=outs[1], LossV=outs[2], QVals=outs[3],
            #              VVals=outs[4], LogPi=outs[5])

            # Update target networks
            self.network.update_target_network()


class SoftActorCritic(BaseAgent):
    def __init__(self, config):
        network_manager = SoftActorCritic_Network_Manager(config)
        super(SoftActorCritic, self).__init__(config, network_manager)





