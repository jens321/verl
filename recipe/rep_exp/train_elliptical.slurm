#!/bin/bash
#SBATCH --job-name=train_elliptical
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=90
#SBATCH --mem=900G
#SBATCH --time=00-23:59:59    # Run for 23:59:59 hours
#SBATCH --gres=gpu:8
#SBATCH --partition=pli-c
#SBATCH --account=pli

set -x

TASK=${1}
SPARSE_DIM=${2}
BETA=${3}
SEED=${4}

sh train_elliptical.sh $TASK $SPARSE_DIM $BETA $SEED

exit 0