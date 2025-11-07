TASK=math
MODEL_PATH=''
sbatch --job-name=eval_${TASK} recipe/rep_exp/eval.slurm ${TASK} ${MODEL_PATH}