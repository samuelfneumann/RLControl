#!/usr/bin/env python3


import click
import subprocess
import multiprocessing
import json
from utils.main_utils import get_sweep_parameters

NUM_PROCESSES = 4

@click.command()
# @click.option("--indices", nargs=3, help="Settings indices to run: start step stop", type=int)
@click.option("--runs", type=int, required=True)
@click.option("--env_name", help="Environment name for experiment", type=str, required=True)
@click.option("--agent_name", help="Filename (no extension) for json file of agent to run for experiment", type=str, required=True)
@click.option("--num_processes", default=NUM_PROCESSES, help="The max number of concurrent processes")
def run(env_name, agent_name, num_processes, runs):
    with open("jsonfiles/agent/" + agent_name + ".json") as in_file:
        agent_json = json.load(in_file)
    _, total_num_sweeps = get_sweep_parameters(agent_json['sweeps'], 0)

    # if indices is None:
    #     indices = [0, 1, runs * total_num_sweeps]

    args = []
    for i in range(total_num_sweeps):
        arg = (f"jsonfiles/environment/{env_name}.json",
               f"jsonfiles/agent/{agent_name}.json", i, total_num_sweeps, total_num_sweeps * runs)
        args.append(arg)

    with multiprocessing.Pool(num_processes) as p:
            p.starmap(run_experiment, args)

def run_experiment(env_file, agent_file, start, step, stop):
    subprocess.run(["python3", "main.py", "--env_json", env_file,
                    "--agent_json", agent_file, "--indices", str(start),
                    str(step), str(stop)])

if __name__ == "__main__":
    run()
