#!/usr/bin/env python3

# Import modules
import sys 
sys.path.insert(1, "/home/sfneuman/Actor-Expert/RLControl")
import click
import subprocess
import multiprocessing
import os
import json
from glob import glob
import pickle
from utils.main_utils import get_sweep_parameters

# TODO:
#   Read in the json files once and pass to subprocesses


NUM_PROCESSES = 1

@click.command()
# @click.option("--indices", nargs=3, help="Settings indices to run: start step stop", type=int)
@click.option("--runs", type=int, required=True)
@click.option("--env_name", help="Environment name for experiment",
              type=str, required=True)
@click.option("--agent_name", help="Filename (no extension) for json file " +
              "of agent to run for experiment", type=str, required=True)
@click.option("--num_processes", default=NUM_PROCESSES,
              help="The max number of concurrent processes")
@click.option("--save_dir", default="./results")
def run(env_name, agent_name, num_processes, runs, save_dir):
    """
    Runs concurrent or serial experiments

    This function runs experiments based on the hyperparameter settings
    determined in the JSON configuration files. This function will look in
    ./results to find the configuration files and load them in. Each distinct
    hyperparameter setting is run serially, and concurrent runs are only
    performed between different hyperparameter settings.

    Parameters
    ----------
    env_name : str
        The name of the environment JSON file, without the file extension
    agent_name : str
        The name of the agent JSON file, without the file extension
    num_processes : int
        Maximum number of concurrent processes
    runs : int
        The number of runs per experiment
    """
    # Create save directory before running concurrently
    data_path = save_dir + "/" + env_name + "_" + \
        agent_name + 'results/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    with open("/home/sfneuman/Actor-Expert/RLControl/jsonfiles/agent/" + agent_name + ".json") as in_file:
        agent_json = json.load(in_file)
    _, total_num_sweeps = get_sweep_parameters(agent_json['sweeps'], 0)

    agent_file = f"/home/sfneuman/Actor-Expert/RLControl/jsonfiles/agent/{agent_name}.json"
    env_file = f"/home/sfneuman/Actor-Expert/RLControl/jsonfiles/environment/{env_name}.json"

    if num_processes != 1:
        # Concurrent runs
        args = []
        for i in range(total_num_sweeps):
            arg = (env_file, agent_file, i, total_num_sweeps,
                   total_num_sweeps * runs, save_dir)
            args.append(arg)

        with multiprocessing.Pool(num_processes) as p:
                p.starmap(run_experiment, args)

        # Combine data files
        #data_path=os.path.join(save_dir, f"{env_name}_{agent_name}results")
        combine_data_dictionaries(data_path)
    else:
        # Sequential runs
        run_experiment(env_file, agent_file, 0, 1, total_num_sweeps * runs,
                       save_dir)



def run_experiment(env_file, agent_file, start, step, stop, save_dir):
    """
    Runs a single experiment

    Parameters
    ----------
    env_file : str
        Path to the environment configuration file
    agent_file : str
        Path to the agent configuration file
    start : int
        Starting hyperparameter index for the experiment
    step : int
        Step value for the hyperparameter index
    stop : int
        The hyperparameter index to stop the experiment at
    """
    subprocess.run(["python3", "/home/sfneuman/Actor-Expert/RLControl/main.py", "--env_json", env_file,
                    "--agent_json", agent_file, "--indices", str(start),
                    str(step), str(stop), "--save_dir", save_dir])


def combine_data_dictionaries(dir):
    """
    Combines the many data dictionaries created during the concurrent
    training procedure into a single data dictionary. The combined data is
    saved as "data.pkl" in the argument dir.

    Parameters
    ----------
    dir : str
        The path to the directory containing all data dictionaries to combine

    Returns
    -------
    dict
        The combined dictionary

    Raises
    ------
    KeyError
        If one dictionary would overwrite another due to the same key
        existing in both dictionaries
    """
    print(dir)
    print(os.listdir(dir))
    files = glob(os.path.join(dir, "*.pkl"))
    print(files)

    # Use first dictionary as base dictionary
    with open(files[0], "rb") as in_file:
        data = pickle.load(in_file)

    # Add data from all other dictionaries
    for file in files[1:]:
        with open(file, "rb") as in_file:
            # Read in the new dictionary
            in_data = pickle.load(in_file)

            # Add experiment data to running dictionary
            for key in in_data["experiment_data"]:
                # Check if key exists
                if key in data["experiment_data"]:
                    print(key)
                    raise KeyError("cannot add key that already exists")

                # Add data to dictionary
                data["experiment_data"][key] = in_data["experiment_data"][key]

    with open(os.path.join(dir, "data.pkl"), "wb") as out_file:
        pickle.dump(data, out_file)

    return data


if __name__ == "__main__":
    run()
