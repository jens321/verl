TASK=countdown-4
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
RESUME_MODE=resume_path
RESUME_FROM_PATH='checkpoints/llm-exploration-rl-training/countdown-4_elliptical_seed_41_kl_0.0_ppo_epochs_1_beta_0.01/best_pass@1'
CHECKPOINT_SAVE_CONTENTS='["model"]'

echo "Eval job on ${TASK} with the following parameters:"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "RESUME_MODE: ${RESUME_MODE}"
echo "RESUME_FROM_PATH: ${RESUME_FROM_PATH}"
echo "CHECKPOINT_SAVE_CONTENTS: ${CHECKPOINT_SAVE_CONTENTS}"
# sbatch --job-name=eval_${TASK} scripts/eval/eval.slurm \
sh scripts/eval/eval.slurm \
    ${MODEL_PATH} \
    ${RESUME_MODE} \
    "${RESUME_FROM_PATH}" \
    ${TASK} \
    ${CHECKPOINT_SAVE_CONTENTS}
echo "--------------------------------"
