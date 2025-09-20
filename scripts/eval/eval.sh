TASK=countdown-4
# NAME=math_elliptical_seed_42_kl_0.0_ppo_epochs_1_beta_0.01_turn_off_elliptical_if_none_correct_True_sparse_dim_32/best_pass@1/actor/hf
# NAME=math_grpo_seed_43_kl_0.0_ppo_epochs_1/best_pass@1/actor/hf
# NAME=math_unlikely_seed_43_kl_0.0/best_pass@1/actor/hf

# NAME=gsm8k_elliptical_seed_43_kl_0.0_ppo_epochs_1_beta_0.01_turn_off_elliptical_if_none_correct_True_sparse_dim_32/best_pass@1/actor/hf
# NAME=gsm8k_grpo_seed_43_kl_0.0_ppo_epochs_1/best_pass@1/actor/hf
# NAME=gsm8k_unlikely_seed_43_kl_0.0/best_pass@1/actor/hf

# COUNTDOWN-4
# NAME=countdown-4_elliptical_seed_41_kl_0.0_ppo_epochs_1_beta_0.01_turn_off_elliptical_if_none_correct_True_sparse_dim_128/global_step_160/actor/hf
NAME=countdown-4_elliptical_seed_43_kl_0.0_ppo_epochs_1_beta_0.01_turn_off_elliptical_if_none_correct_True_sparse_dim_128/global_step_200/actor/hf

# NAME=countdown-4_grpo_seed_41_kl_0.0_ppo_epochs_1/global_step_360/actor/hf
# NAME=countdown-4_grpo_seed_43_kl_0.0_ppo_epochs_1/global_step_300/actor/hf

# AIME
# NAME=dapo-with-aime2425_elliptical_seed_41_kl_0.0_ppo_epochs_1_beta_0.01_turn_off_elliptical_if_none_correct_True_sparse_dim_128/global_step_70/actor/hf
# NAME=dapo-with-aime2425_elliptical_seed_43_kl_0.0_ppo_epochs_1_beta_0.01_turn_off_elliptical_if_none_correct_True_sparse_dim_128/global_step_100/actor/hf

# NAME=dapo-with-aime2425_grpo_seed_41_kl_0.0_ppo_epochs_1/global_step_230/actor/hf

MODEL_PATH=/scratch/gpfs/jtuyls/llm-rl-exploration/checkpoints/llm-exploration-rl-training/${NAME}
# MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
RESUME_MODE=disable
RESUME_FROM_PATH=""
CHECKPOINT_SAVE_CONTENTS='["model"]'

# assert TASK is in RESUME_FROM_PATH
# if ! echo "${MODEL_PATH}" | grep -q "${TASK}"; then
#     echo "ERROR: TASK is not in MODEL_PATH"
#     exit 1
# fi

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
