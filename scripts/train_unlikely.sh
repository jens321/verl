TASK=dapo-with-aime2425 # math, gsm8k, countdown-4, dapo-with-aime2425
ALGORITHM=grpo
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
BETA=0.25
ROLLOUTS=8 # default: 32
TEST_FREQ=20
SAVE_FREQ=20
RESUME_MODE=disable
RESUME_FROM_PATH=''
USE_KL_LOSS=True
SAVE_BEST_PASS_AT_1=False
SAVE_BEST_HARD_PASS_AT_1=False
SAVE_BEST_PASS_AT_64=False
SAVE_BEST_HARD_PASS_AT_64=False
CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra"]'
MAX_ACTOR_CKPT_TO_KEEP=null
TRAIN_BATCH_SIZE=1024 # default: $((256 / ${ROLLOUTS}))
PPO_MINI_BATCH_SIZE=256 # default: $((256 / ${ROLLOUTS}))
DROP_SAMPLES_WITH_NO_ADV=False # default: True
PPO_EPOCHS=1 # default: 2
GEN_BATCH_SIZE=null # default: 16
GRAD_SKIP_THRESH=null # default: 20.0
TURN_OFF_UNLIKELY_IF_ALL_CORRECT=True # default: False
REMOVE_PREVIOUS_OPTIM_AND_EXTRA=True

# GRPO specific
LOSS_AGG_MODE="token-mean"
KL_LOSS_COEF=0.0 # default: 0.1
NORM_ADV_BY_STD_IN_GRPO=True

if [ ${TASK} == "dapo-with-aime2425" ]; then
    TEST_FREQ=10
    SAVE_FREQ=10
    TRAIN_BATCH_SIZE=512
    PPO_MINI_BATCH_SIZE=128
fi

for SEED in 42; do
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
    echo "TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE}"
    echo "PPO_MINI_BATCH_SIZE: ${PPO_MINI_BATCH_SIZE}"
    echo "DROP_SAMPLES_WITH_NO_ADV: ${DROP_SAMPLES_WITH_NO_ADV}"
    echo "PPO_EPOCHS: ${PPO_EPOCHS}"
    echo "GEN_BATCH_SIZE: ${GEN_BATCH_SIZE}"
    echo "GRAD_SKIP_THRESH: ${GRAD_SKIP_THRESH}"
    echo "TURN_OFF_UNLIKELY_IF_ALL_CORRECT: ${TURN_OFF_UNLIKELY_IF_ALL_CORRECT}"
    echo "REMOVE_PREVIOUS_OPTIM_AND_EXTRA: ${REMOVE_PREVIOUS_OPTIM_AND_EXTRA}"
    sbatch --job-name=${TASK}_unlikely_seed_${SEED}_kl_${KL_LOSS_COEF} scripts/train_unlikely.slurm \
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
        ${MAX_ACTOR_CKPT_TO_KEEP} \
        ${TRAIN_BATCH_SIZE} \
        ${PPO_MINI_BATCH_SIZE} \
        ${DROP_SAMPLES_WITH_NO_ADV} \
        ${PPO_EPOCHS} \
        ${GEN_BATCH_SIZE} \
        ${GRAD_SKIP_THRESH} \
        ${TURN_OFF_UNLIKELY_IF_ALL_CORRECT} \
        ${REMOVE_PREVIOUS_OPTIM_AND_EXTRA}
    echo "--------------------------------"
done