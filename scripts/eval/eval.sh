TASK=math
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
RESUME_MODE=resume_path
RESUME_FROM_PATH='/scratch/gpfs/KARTHIKN/jtuyls/checkpoints/llm-exploration-rl-training/math_unlikely_seed_43_kl_0.0/best_pass@1'
CHECKPOINT_SAVE_CONTENTS='["model"]'

# assert TASK is in RESUME_FROM_PATH
if ! echo "${RESUME_FROM_PATH}" | grep -q "${TASK}"; then
    echo "ERROR: TASK is not in RESUME_FROM_PATH"
    exit 1
fi

echo "Eval job on ${TASK} with the following parameters:"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "RESUME_MODE: ${RESUME_MODE}"
echo "RESUME_FROM_PATH: ${RESUME_FROM_PATH}"
echo "CHECKPOINT_SAVE_CONTENTS: ${CHECKPOINT_SAVE_CONTENTS}"
sbatch --job-name=eval_${TASK} scripts/eval/eval.slurm \
    ${MODEL_PATH} \
    ${RESUME_MODE} \
    "${RESUME_FROM_PATH}" \
    ${TASK} \
    ${CHECKPOINT_SAVE_CONTENTS}
echo "--------------------------------"
