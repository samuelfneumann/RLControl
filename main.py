# -*- encoding:utf8 -*-
import gym
import tensorflow as tf
import environments.environments as envs
from utils.config import Config
from os.path import basename

import numpy as np
import pickle
import json
import os
import datetime
from collections import OrderedDict

import argparse
import subprocess

# TODO:
#   Change to run multiple indices
#   Change to save a dictionary
#       Then, we can save, for each HP setting, a specific dictionary of all
#       data (the 5 files currently saved)
#       This way, we will save a single pickled object, and not multiple small files


# @click.command()
# @click.option("--env_json", type=str, help="The json file of the environment")
# @click.option("--agent_json", type=str, help="The json file of the agent")
# @click.option("--index", type=int, help="The settings index to run")
# @click.option("--monitor", type=bool, default=False, )
def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_json', type=str)
    parser.add_argument('--agent_json', type=str)
    parser.add_argument('--indices', type=int, nargs=3)
    parser.add_argument('--monitor', default=False, action='store_true')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--write_log', default=False, action='store_true')
    parser.add_argument('--write_plot', default=False, action='store_true')
    parser.add_argument('--save_dir', default="./results")

    args = parser.parse_args()

    arg_params = {
        "write_log": args.write_log,
        "write_plot": args.write_plot
    }

    # Get name of env and agent for save directory so it is consistent with
    # command line arguments
    env_name = args.env_json
    agent_name = args.agent_json

    # read env/agent json
    with open(args.env_json, 'r') as env_dat:
        env_json = json.load(env_dat, object_pairs_hook=OrderedDict)

    with open(args.agent_json, 'r') as agent_dat:
        agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)

    # initialize env
    train_env = envs.create_environment(env_json)
    test_env = envs.create_environment(env_json)

    # Create env_params for agent
    env_params = {
            "env_name": train_env.name,
            "state_dim": train_env.state_dim,
            "state_min": train_env.state_min,
            "state_max": train_env.state_max,

            "action_dim": train_env.action_dim,
            "action_min": train_env.action_min,
            "action_max": train_env.action_max
    }

    # Create data dictionary and store relevant information
    data = {}
    data["experiment"] = {}

    # Experiment meta-data
    data["experiment"]["environment"] = {}
    data["experiment"]["agent"] = {}
    data["experiment"]["agent"]["agent_name"] = agent_json["agent"]
    data["experiment"]["agent"]["parameters"] = dict(agent_json["sweeps"])
    data["experiment"]["environment"]["env_name"] = env_json["environment"]
    data["experiment"]["environment"]["total_timesteps"] = env_json["TotalMilSteps"] * 1000000
    data["experiment"]["environment"]["steps_per_episode"] = env_json["EpisodeSteps"]
    data["experiment"]["environment"]["eval_interval_timesteps"] = env_json["EvalIntervalMilSteps"] * 1000000
    data["experiment"]["environment"]["eval_episodes"] = env_json["EvalEpisodes"]

    # Experiment runs per each hyperparameter
    data["experiment_data"] = {}

    from utils.main_utils import get_sweep_parameters, create_agent
    for index in range(args.indices[0], args.indices[2], args.indices[1]):

        agent_params, total_num_sweeps = get_sweep_parameters(agent_json['sweeps'], index)
        print(f"Total HP settings: {total_num_sweeps}")

        sweep = index % total_num_sweeps
        if sweep not in data["experiment_data"].keys():
            data["experiment_data"][sweep] = {}
            data["experiment_data"][sweep]["agent_params"] = dict(agent_params)
            data["experiment_data"][sweep]["runs"] = []
            # All the following are arrays of arrays, where the nested array
            # is the run number
            # data["hp_settings"]["runs"]["eval_episode_rewards"] = []
            # data["experiment_data"][sweep]["runs"]["run_number"] = []
            # data["hp_settings"]["runs"]["random_seeds"] = []
            # data["hp_settings"]["runs"]["eval_episode_std_rewards"] = []
            # data["hp_settings"]["runs"]["eval_episode_mean_rewards"] = []
            # data["hp_settings"]["runs"]["episode_steps"] = []
            # data["hp_settings"]["runs"]["online_episode_rewards"] = []

        # get run idx and setting idx
        run_data = {}
        RUN_NUM = int(index / total_num_sweeps)
        # run_data["run_number"] = RUN_NUM

        SETTING_NUM = index % total_num_sweeps

        # set Random Seed (for training)
        RANDOM_SEED = RUN_NUM
        arg_params['random_seed'] = RANDOM_SEED
        run_data["random_seed"] = RANDOM_SEED

        print(f"SETTING_NUM: {SETTING_NUM}")
        print(f"RUN_NUM: {RUN_NUM}")
        print(f"RANDOM_SEED: {RANDOM_SEED}")
        print('Agent setting: ', agent_params)

        # create save directory
        save_dir = args.save_dir + "/" + env_name + "_" + \
            agent_name + 'results/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # create log directory (for tensorboard, gym monitor/render)
        START_DATETIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        env_name = env_json["environment"]
        agent_name = agent_json["agent"]
        log_dir = './results/{}_{}results/log_summary/{}/{}_{}_{}'.format(str(env_json['environment']), str(agent_json['agent']), str(agent_json['agent']), str(SETTING_NUM), str(RUN_NUM), str(START_DATETIME))

        # tf 1.8
        #writer = tf.summary.create_file_writer(log_dir)
        writer = tf.summary.FileWriter(log_dir)
        # tf 2.0
        # writer = tf.summary.create_file_writer(log_dir)
        agent_params["writer"] = writer

        # init config and merge custom config settings from json
        config = Config()
        config.merge_config(env_params)
        config.merge_config(agent_params)
        config.merge_config(arg_params)

        # initialize agent
        agent = create_agent(agent_json['agent'], config)

        # monitor/render
        if args.monitor or args.render:
            monitor_dir = log_dir+'/monitor'

            if args.render:
                train_env.instance = gym.wrappers.Monitor(train_env.instance, monitor_dir, video_callable=(lambda x: True), force=True)
            else:
                train_env.instance = gym.wrappers.Monitor(train_env.instance, monitor_dir, video_callable=False, force=True)

        # initialize experiment
        from experiment import Experiment
        experiment = Experiment(agent=agent, train_environment=train_env, test_environment=test_env, seed=RANDOM_SEED,
                                writer=writer, write_log=args.write_log, write_plot=args.write_plot)

        # run experiment
        episode_rewards, eval_episode_rewards, train_episode_steps, \
            eval_episode_steps, timesteps_at_eval, train_time, eval_time, \
            train_ep, _ = experiment.run()

        # Save data
        run_data["total_timesteps"] = env_json["TotalMilSteps"] * 1000000
        run_data["eval_interval_timesteps"] = env_json["EvalIntervalMilSteps"] * 1000000
        run_data["episodes_per_eval"] = env_json["EvalEpisodes"]

        # run_data["eval_episode_std_rewards"] = np.array(eval_episode_std_rewards)
        # run_data["eval_episode_mean_rewards"] = np.array(eval_episode_mean_rewards)
        run_data["eval_episode_rewards"] = np.array(eval_episode_rewards)
        run_data["eval_episode_steps"] = np.array(eval_episode_steps)
        run_data["timesteps_at_eval"] = np.array(timesteps_at_eval)
        run_data["train_episode_steps"] = np.array(train_episode_steps)
        run_data["train_episode_rewards"] = np.array(episode_rewards)
        run_data["total_train_episodes"] = train_ep
        run_data["eval_time"] = train_time
        run_data["train_time"] = eval_time

        data["experiment_data"][sweep]["runs"].append(run_data)

        save_file = save_dir + env_json['environment'] + '_' + \
            agent_json['agent'] + f"_data_{args.indices[0]}_{args.indices[1]}_{args.indices[2]}.pkl"
        with open(save_file, "wb") as out_file:
            pickle.dump(data, out_file)
            print(data)

        # save to file
        # prefix = save_dir + env_json['environment'] + '_'+agent_json['agent'] + '_setting_' + str(SETTING_NUM) + '_run_'+str(RUN_NUM)

        # train_rewards_filename = prefix + '_EpisodeRewardsLC.txt'
        # np.array(episode_rewards).tofile(train_rewards_filename, sep=',', format='%15.8f')

        # eval_mean_rewards_filename = prefix + '_EvalEpisodeMeanRewardsLC.txt'
        # np.array(eval_episode_mean_rewards).tofile(eval_mean_rewards_filename, sep=',', format='%15.8f')

        # eval_std_rewards_filename = prefix + '_EvalEpisodeStdRewardsLC.txt'
        # np.array(eval_episode_std_rewards).tofile(eval_std_rewards_filename, sep=',', format='%15.8f')

        # train_episode_steps_filename = prefix + '_EpisodeStepsLC.txt'
        # np.array(train_episode_steps).tofile(train_episode_steps_filename, sep=',', format='%15.8f')

        # params = []
        # # params_names = '_'
        # for key in agent_params:
        #     # for Python 2 since JSON load delivers "unicode" rather than pure string
        #     # then it will produce problem at plotting stage
        #     if isinstance(agent_params[key], type(u'')):
        #         params.append(agent_params[key].encode('utf-8'))
        #     else:
        #         params.append(agent_params[key])
        #     # params_names += (key + '_')

        # params = np.array(params)
        # # name = prefix + params_names + 'Params.txt'
        # name = prefix + '_agent_' + 'Params.txt'
        # params.tofile(name, sep=',', format='%s')

        # save json file as well
        # Bimodal1DEnv_uneq_var1_ActorCritic_agent_Params
        with open('{}{}_{}_agent_Params.json'.format(save_dir, env_json['environment'], agent_json['agent']), 'w') as json_save_file:
            json.dump(agent_json, json_save_file)

        # generate video and delete figures
        if args.write_plot:
            subprocess.run(["ffmpeg", "-framerate", "24", "-i", "{}/figures/steps_%01d.png".format(log_dir), "{}.mp4".format(log_dir)])
            # subprocess.run(["mv", "{}.mp4".format(log_dir), "{}/../".format(log_dir)])
            subprocess.run(["rm", "-rf", "{}/figures".format(log_dir)])


if __name__ == '__main__':
    main()


