TASK=math
ALGORITHM=grpo
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
BETA=0.25
ROLLOUTS=32
TEST_FREQ=20
SAVE_FREQ=-1
RESUME_MODE=disable
RESUME_FROM_PATH=''
USE_KL_LOSS=True
SAVE_BEST_PASS_AT_1=False
SAVE_BEST_HARD_PASS_AT_1=False
SAVE_BEST_PASS_AT_64=False
SAVE_BEST_HARD_PASS_AT_64=False
CHECKPOINT_SAVE_CONTENTS='["model"]'
MAX_ACTOR_CKPT_TO_KEEP=1

# GRPO specific
LOSS_AGG_MODE="token-mean"
KL_LOSS_COEF=0.1
NORM_ADV_BY_STD_IN_GRPO=True

for SEED in 44 45; do
    echo "Running job on ${TASK} with the following parameters:"
    echo "ALGORITHM: ${ALGORITHM}"
    echo "MODEL_PATH: ${MODEL_PATH}"
    echo "SEED: ${SEED}"
    echo "BETA: ${BETA}"
    echo "ROLLOUTS: ${ROLLOUTS}"
    echo "LOSS_AGG_MODE: ${LOSS_AGG_MODE}"
    echo "USE_KL_LOSS: ${USE_KL_LOSS}"
    echo "NORM_ADV_BY_STD_IN_GRPO: ${NORM_ADV_BY_STD_IN_GRPO}"
    echo "TEST_FREQ: ${TEST_FREQ}"
    echo "SAVE_FREQ: ${SAVE_FREQ}"
    echo "RESUME_MODE: ${RESUME_MODE}"
    echo "RESUME_FROM_PATH: ${RESUME_FROM_PATH}"
    echo "KL_LOSS_COEF: ${KL_LOSS_COEF}"
    echo "SAVE_BEST_PASS_AT_1: ${SAVE_BEST_PASS_AT_1}"
    echo "SAVE_BEST_PASS_AT_64: ${SAVE_BEST_PASS_AT_64}"
    echo "CHECKPOINT_SAVE_CONTENTS: ${CHECKPOINT_SAVE_CONTENTS}"
    echo "SAVE_BEST_HARD_PASS_AT_1: ${SAVE_BEST_HARD_PASS_AT_1}"
    echo "SAVE_BEST_HARD_PASS_AT_64: ${SAVE_BEST_HARD_PASS_AT_64}"
    echo "MAX_ACTOR_CKPT_TO_KEEP: ${MAX_ACTOR_CKPT_TO_KEEP}"
    sbatch --job-name=train_${ALGORITHM}_MATH_unlikely_beta_${BETA} scripts/train_unlikely.slurm \
        ${MODEL_PATH} \
        ${SEED} \
        ${BETA} \
        ${ROLLOUTS} \
        ${LOSS_AGG_MODE} \
        ${USE_KL_LOSS} \
        ${NORM_ADV_BY_STD_IN_GRPO} \
        ${ALGORITHM} \
        ${TEST_FREQ} \
        ${SAVE_FREQ} \
        ${RESUME_MODE} \
        "${RESUME_FROM_PATH}" \
        ${KL_LOSS_COEF} \
        ${TASK} \
        ${SAVE_BEST_PASS_AT_1} \
        ${SAVE_BEST_PASS_AT_64} \
        ${CHECKPOINT_SAVE_CONTENTS} \
        ${SAVE_BEST_HARD_PASS_AT_1} \
        ${SAVE_BEST_HARD_PASS_AT_64} \
        ${MAX_ACTOR_CKPT_TO_KEEP}
    echo "--------------------------------"
done
