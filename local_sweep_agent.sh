#!/bin/bash

Help() {
	echo -e "Local Sweep Agent: Sweeps the hyperparameters of the agent\n"
	echo -e "Usage:\t$(basename $0) ENV_NAME AGENT_NAME start_idx increment end_idx"
	echo -e "\tstart_idx, end_idx are the inclusive settings combination indices as defined in the agent JSON file"
	echo -e "\tENV_NAME and AGENT_NAME are the names of the environment and agent, and not the paths to the JSON files"
	echo -e "\nArguments"
	echo -e "\t-h \t help"
}


while getopts ":h" option; do
	case $option in
		h)  # display help
			Help
			shift $((OPTIND -1))
			exit;;
	esac
done


ENV_NAME=$1
AGENT_NAME=$2

source /home/samuel/Documents/Actor-Expert/actor_expert_env/bin/activate
#echo "Bash version ${BASH_VERSION}..."

# Inclusive
start_idx=$3
increment=$4
end_idx=$5
for i in $(seq ${start_idx} ${increment} ${end_idx})
do
	echo "====================================================="
	echo -e "\nRunning Experiment: $i\n"
	echo "====================================================="
	python3 main.py --env_json jsonfiles/environment/"$ENV_NAME".json --agent_json jsonfiles/agent/"$AGENT_NAME".json --index "$i" # --write_plot
done
