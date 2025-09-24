TASK=dapo-with-aime2425 # math, gsm8k, countdown-4, dapo-with-aime2425
ALGORITHM=grpo
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
ROLLOUTS=8
TEST_FREQ=20
SAVE_FREQ=20
RESUME_MODE=resume_path # resume_path, disable
RESUME_FROM_PATH='/scratch/gpfs/PLI/jtuyls/checkpoints/llm-exploration-rl-training/dapo-with-aime2425_grpo_seed_43_kl_0.0_ppo_epochs_1/global_step_210'
USE_KL_LOSS=True
PPO_EPOCHS=1
SAVE_BEST_PASS_AT_1=False
SAVE_BEST_HARD_PASS_AT_1=False
SAVE_BEST_PASS_AT_64=False
SAVE_BEST_HARD_PASS_AT_64=False
CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra"]'
MAX_ACTOR_CKPT_TO_KEEP=null
TRAIN_BATCH_SIZE=1024
PPO_MINI_BATCH_SIZE=256
REMOVE_PREVIOUS_OPTIM_AND_EXTRA=True

if [ ${ALGORITHM} == "dr_grpo" ]; then
    LOSS_AGG_MODE="seq-mean-token-sum-norm"
    KL_LOSS_COEF=0.0
    NORM_ADV_BY_STD_IN_GRPO=False
else
    LOSS_AGG_MODE="token-mean"
    KL_LOSS_COEF=0.0 # default: 0.001
    NORM_ADV_BY_STD_IN_GRPO=True
fi

if [ ${TASK} == "dapo-with-aime2425" ]; then
    TEST_FREQ=10
    SAVE_FREQ=10
    TRAIN_BATCH_SIZE=512
    PPO_MINI_BATCH_SIZE=128
fi

for SEED in 43; do
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
    echo "TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE}"
    echo "PPO_MINI_BATCH_SIZE: ${PPO_MINI_BATCH_SIZE}"
    echo "REMOVE_PREVIOUS_OPTIM_AND_EXTRA: ${REMOVE_PREVIOUS_OPTIM_AND_EXTRA}"
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
        ${MAX_ACTOR_CKPT_TO_KEEP} \
        ${TRAIN_BATCH_SIZE} \
        ${PPO_MINI_BATCH_SIZE} \
        ${REMOVE_PREVIOUS_OPTIM_AND_EXTRA}
    echo "--------------------------------"
done
