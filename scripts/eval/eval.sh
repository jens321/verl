TASK=math
# NAME=math_elliptical_seed_42_kl_0.0_ppo_epochs_1_beta_0.01_turn_off_elliptical_if_none_correct_True_sparse_dim_32
# NAME=math_grpo_seed_43_kl_0.0_ppo_epochs_1
NAME=math_unlikely_seed_43_kl_0.0
MODEL_PATH=/scratch/gpfs/KARTHIKN/jtuyls/checkpoints/llm-exploration-rl-training/${NAME}/best_pass@1/actor/hf #Qwen/Qwen2.5-7B-Instruct
RESUME_MODE=disable #resume_path
RESUME_FROM_PATH="" #'/scratch/gpfs/KARTHIKN/jtuyls/checkpoints/llm-exploration-rl-training/math_unlikely_seed_43_kl_0.0/best_pass@1'
CHECKPOINT_SAVE_CONTENTS='["model"]'

# assert TASK is in RESUME_FROM_PATH
if ! echo "${MODEL_PATH}" | grep -q "${TASK}"; then
    echo "ERROR: TASK is not in MODEL_PATH"
    exit 1
fi

echo "Eval job on ${TASK} with the following parameters:"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "RESUME_MODE: ${RESUME_MODE}"
echo "RESUME_FROM_PATH: ${RESUME_FROM_PATH}"
echo "CHECKPOINT_SAVE_CONTENTS: ${CHECKPOINT_SAVE_CONTENTS}"
sbatch --job-name=eval_${NAME} scripts/eval/eval.slurm \
    ${MODEL_PATH} \
    ${RESUME_MODE} \
    "${RESUME_FROM_PATH}" \
    ${TASK} \
    ${CHECKPOINT_SAVE_CONTENTS}
echo "--------------------------------"
