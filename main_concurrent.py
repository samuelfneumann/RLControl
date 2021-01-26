#!/usr/bin/env python3


import click
import subprocess
import multiprocessing


@click.command()
@click.option("--indices", nargs=3, help="Settings indices to run: start step stop", type=int)
@click.option("--env_name", help="Environment name for experiment", type=str)
@click.option("--agent_name", help="Filename (no extension) for json file of agent to run for experiment", type=str)
def run(indices, env_name, agent_name):
    args = []
    for i in range(indices[0], indices[2], indices[1]):
        arg = (f"jsonfiles/environment/{env_name}.json",
               f"jsonfiles/agent/{agent_name}.json", i)
        args.append(arg)

    with multiprocessing.Pool(2) as p:
            p.starmap(run_experiment, args)

def run_experiment(env_file, agent_file, index):
    subprocess.run(["python3", "main.py", "--env_json", env_file,
                    "--agent_json", agent_file, "--index", str(index)])

if __name__ == "__main__":
    run()
