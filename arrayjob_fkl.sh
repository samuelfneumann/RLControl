#!/bin/bash
#
# Runs the ForwardKL experiments

#SBATCH --job-name="ArrayForwardKL"
#SBATCH --account=rrg-whitem

# Number of cores and memory in MiB
#SBATCH --array=0-4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000

# Time for job completion
#SBATCH -t 0-00:10:00

# Warn before terminating on timeout
#SBATCH --signal=B:USR1@120

# Variables used in the script to run experiments
JOB_NUM="$SLURM_ARRAY_TASK_ID"
PROCESSES=1  #${SLURM_CPUS_PER_TASK}
AGENT="forward_kl"
ENV="Pendulum-v0"
RUNS=1
SOURCE_DIR="/home/sfneuman/Actor-Expert/RLControl"
SAVE_DIR="$SOURCE_DIR/results/""$ENV""_$AGENT""results"
TEMP_DATA_DIR="$SLURM_TMPDIR/data"
CLEANUP_CALLED=false


################################
# Cleans up files
#
# Moves the important experiment data files from a temporary directory
# to a non-temporary directory
# Globals:
#	CLEANUP_CALLED - whether or not the cleanup function has already
#			 been called
#	SOURCE_DIR - the directory which contains all source code
#	TEMP_DATA_DIR - the temporary directory that the data is saved in
#	AGENT - the name of the agent trained
#	ENV - the name of the environment trained on
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

	#local cwd=$(pwd)
	#cd "$TEMP_DATA_DIR"
	#local tarball_results="$ENV""_$AGENT"_"$JOB_NUM".tar.gz
	#tar -cvzf "$tarball_results" "$ENV""_""$AGENT""results"
	#cp "$tarball_results" "$SAVE_DIR"
	#cd "$cwd"
	cp -a "$TEMP_DATA_DIR""/$ENV""_$AGENT""results/." "$SAVE_DIR"

	CLEANUP_CALLED=true
	#rm -r $TEMP_DATA_DIR
	#rm -r $SLURM_TMPDIR/env
}


# Save if job is to be terminated
trap cleanup SIGUSR1 SIGINT SIGTERM SIGKILL

# Load in required modules
module load StdEnv/2020 cudacore/.11.0.2
module load cuda/11.0
module load cudnn/8.0.3
module load python/3.7

# Prepare virutal env
echo "creating $(python3 --version) venv"
python3 -m virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install requirements
echo "installing requirements..."
pip install numpy scipy quadpy spinup matplotlib click joblib scikit_learn pillow==7.2.0 gym
pip install --no-index tensorflow_cpu==1.15.0 torch==1.7.1
echo -e "done\n"

# Prepare data directory
mkdir "$TEMP_DATA_DIR"
mkdir "$TEMP_DATA_DIR/""$ENV""_$AGENT""results"
if [[ ! -d "$SAVE_DIR" ]]; then
	echo "Creating save directory $SAVE_DIR"
	mkdir "$SAVE_DIR"
fi

echo $TEMP_DATA_DIR
ls $TEMP_DATA_DIR

# Train
$SOURCE_DIR/local_sweep_agent.sh "$ENV" "$AGENT" $SLURM_ARRAY_TASK_ID 1 $(( $SLURM_ARRAY_TASK_ID + 1 )) $TEMP_DATA_DIR

# Clean up, moving the data to a permanent directory
cleanup
