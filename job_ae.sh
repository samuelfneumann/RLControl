#!/bin/bash
#
# Runs the ForwardKL experiments

#SBATCH --job-name="ForwardKL"
#SBATCH --account=rrg-whitem

# Email notification
#SBATCH --mail-user=sfneuman@ualberta.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Number of cores and memory in MiB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000

# Time for job completion
#SBATCH -t 0-00:05

# Warn before terminating on timeout
#SBATCH --signal=B:USR1@120

# Variables used in the script to run experiments
SLURM_TMPDIR="/home/samuel/tmp"
PROCESSES=2  #${SLURM_CPUS_PER_TASK}
AGENT="forward_kl"
ENV="Pendulum-v0"
RUNS=1
SOURCE_DIR="/home/sfneuman/Actor-Expert/RLControl"
SAVE_DIR="$SOURCE_DIR/results"
TEMP_DIR="$SLURM_TMPDIR/data"
CLEANUP_CALLED=false


################################
# Cleans up files
#
# Moves the important experiment data files from a temporary directory
# to a non-temporary directory
# Globals:
#	CLEANUP_CALLED - whether or not the cleanup function has already
#			 been called
################################
cleanup()
{
	# Ensure cleanup is only called once
	if "$CLEANUP_CALLED"; then
		return
	fi

	if [[ ! -d "$SOURCE_DIR"/results ]]; then
		mkdir "$SOURCE_DIR"/results
	fi

	local cwd=$(pwd)
	cd "$TEMP_DIR"
	local tarball_results="$ENV""_$AGENT".tar.gz
	tar -cvzf "$tarball_results" "$ENV""_""$AGENT""results"
	cp "$tarball_results" "$SAVE_DIR"
	cd "$cwd"
	#cp -r "$TEMP_DIR""/$ENV""_""$AGENT""results" "$SAVE_DIR"

	CLEANUP_CALLED=true
	rm -r $TEMP_DIR
}


# Save if job is to be terminated
trap cleanup USR1 SIGINT SIGTERM SIGKILL

# Load in required modules
#module load python/3.6.3

# Prepare virutal env
#virtualenv --no-download $SLURM_TMPDIR/env
#source $SLURM_TMPDIR/env/bin/activate
#pip install --no-index -r "$SOURCE_DIR"/requirements.txt

# Prepare data directory
mkdir "$TEMP_DIR"

# Train
cd $SOURCE_DIR
source $SOURCE_DIR/../actor_expert_env/bin/activate
python "$SOURCE_DIR/"main_concurrent.py --env_name "$ENV" --agent_name "$AGENT" --runs "$RUNS" --save_dir "$TEMP_DIR" --num_processes "$PROCESSES"

echo
echo $TEMP_DIR
ls $TEMP_DIR
echo
ls $SAVE_DIR

# Clean up, moving the data to a permanent directory
cleanup
