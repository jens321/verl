#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=00-23:59:59    # Run for 23:59:59 hours
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

set -x

TASK=${1}
MODEL_PATH=${2}

sh eval.sh ${TASK} ${MODEL_PATH}

exit 0