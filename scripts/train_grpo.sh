TASK=math
ALGORITHM=grpo
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
ROLLOUTS=8
TEST_FREQ=20
SAVE_FREQ=20
RESUME_MODE=disable
RESUME_FROM_PATH=''
USE_KL_LOSS=True
PPO_EPOCHS=2
SAVE_BEST_PASS_AT_1=True
SAVE_BEST_HARD_PASS_AT_1=True
SAVE_BEST_PASS_AT_64=True
SAVE_BEST_HARD_PASS_AT_64=True
CHECKPOINT_SAVE_CONTENTS='["model"]'
MAX_ACTOR_CKPT_TO_KEEP=1

if [ ${ALGORITHM} == "dr_grpo" ]; then
    LOSS_AGG_MODE="seq-mean-token-sum-norm"
    KL_LOSS_COEF=0.0
    NORM_ADV_BY_STD_IN_GRPO=False
else
    LOSS_AGG_MODE="token-mean"
    KL_LOSS_COEF=0.1 # default: 0.001
    NORM_ADV_BY_STD_IN_GRPO=True
fi

for SEED in 42 44 45; do
    echo "Running job on ${TASK} with the following parameters:"
    echo "ALGORITHM: ${ALGORITHM}"
    echo "MODEL_PATH: ${MODEL_PATH}"
    echo "SEED: ${SEED}"
    echo "ROLLOUTS: ${ROLLOUTS}"
    echo "LOSS_AGG_MODE: ${LOSS_AGG_MODE}"
    echo "USE_KL_LOSS: ${USE_KL_LOSS}"
    echo "NORM_ADV_BY_STD_IN_GRPO: ${NORM_ADV_BY_STD_IN_GRPO}"
    echo "TEST_FREQ: ${TEST_FREQ}"
    echo "SAVE_FREQ: ${SAVE_FREQ}"
    echo "RESUME_MODE: ${RESUME_MODE}"
    echo "RESUME_FROM_PATH: ${RESUME_FROM_PATH}"
    echo "KL_LOSS_COEF: ${KL_LOSS_COEF}"
    echo "PPO_EPOCHS: ${PPO_EPOCHS}"
    echo "SAVE_BEST_PASS_AT_1: ${SAVE_BEST_PASS_AT_1}"
    echo "SAVE_BEST_PASS_AT_64: ${SAVE_BEST_PASS_AT_64}"
    echo "CHECKPOINT_SAVE_CONTENTS: ${CHECKPOINT_SAVE_CONTENTS}"
    echo "SAVE_BEST_HARD_PASS_AT_1: ${SAVE_BEST_HARD_PASS_AT_1}"
    echo "SAVE_BEST_HARD_PASS_AT_64: ${SAVE_BEST_HARD_PASS_AT_64}"
    echo "MAX_ACTOR_CKPT_TO_KEEP: ${MAX_ACTOR_CKPT_TO_KEEP}"
    sbatch --job-name=${TASK}_GRPO_seed_${SEED}_kl_${KL_LOSS_COEF}_ppo_epochs_${PPO_EPOCHS} scripts/train_grpo.slurm \
        ${MODEL_PATH} \
        ${SEED} \
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
        ${PPO_EPOCHS} \
        ${SAVE_BEST_PASS_AT_1} \
        ${SAVE_BEST_PASS_AT_64} \
        ${CHECKPOINT_SAVE_CONTENTS} \
        ${SAVE_BEST_HARD_PASS_AT_1} \
        ${SAVE_BEST_HARD_PASS_AT_64} \
        ${MAX_ACTOR_CKPT_TO_KEEP}
    echo "--------------------------------"
done
