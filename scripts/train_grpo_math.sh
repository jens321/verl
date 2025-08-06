ALGORITHM=grpo
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
SPARSE_DIM=8
BETA=0.01
ROLLOUTS=8
REWARD_TYPE=leave_one_out
RANDOMIZE_SPARSE_MATRIX=True
TURN_OFF_ELLIPTICAL_IF_ANY_CORRECT=False
TURN_OFF_ELLIPTICAL_IF_ALL_CORRECT=False
TURN_OFF_AT_HIGHEST_PASS_AT_K=False
TRAIN_RANDOM_SUBSET_SIZE=512
TRAIN_VAL_N=$((2 * ${ROLLOUTS})) # always double the rollout size since we're estimating pass@k where k is the rollout size
ALPHA=1.0
TEST_FREQ=5

if [ ${ALGORITHM} == "dr_grpo" ]; then
    LOSS_AGG_MODE="seq-mean-token-sum-norm"
    USE_KL_LOSS=False
    NORM_ADV_BY_STD_IN_GRPO=False
else
    LOSS_AGG_MODE="token-mean"
    USE_KL_LOSS=True
    NORM_ADV_BY_STD_IN_GRPO=True
fi

if [ ${TURN_OFF_AT_HIGHEST_PASS_AT_K} == True ]; then
    PASS_AT_K_FREQ=5
else
    PASS_AT_K_FREQ=-1
fi

for SEED in 41 42 43; do
    for REWARD_MODEL_ENABLE in True; do
        ELLIPTICAL_ENABLE=${REWARD_MODEL_ENABLE}

        if [ ${REWARD_MODEL_ENABLE} == True ]; then
            REWARD_MANAGER=elliptical
        else
            REWARD_MANAGER=naive
        fi

        echo "Running job with the following parameters:"
        echo "ALGORITHM: ${ALGORITHM}"
        echo "MODEL_PATH: ${MODEL_PATH}"
        echo "REWARD_MODEL_ENABLE: ${REWARD_MODEL_ENABLE}"
        echo "ELLIPTICAL_ENABLE: ${ELLIPTICAL_ENABLE}"
        echo "SPARSE_DIM: ${SPARSE_DIM}"
        echo "REWARD_MANAGER: ${REWARD_MANAGER}"
        echo "SEED: ${SEED}"
        echo "BETA: ${BETA}"
        echo "ROLLOUTS: ${ROLLOUTS}"
        echo "REWARD_TYPE: ${REWARD_TYPE}"
        echo "RANDOMIZE_SPARSE_MATRIX: ${RANDOMIZE_SPARSE_MATRIX}"
        echo "TURN_OFF_ELLIPTICAL_IF_ANY_CORRECT: ${TURN_OFF_ELLIPTICAL_IF_ANY_CORRECT}"
        echo "TURN_OFF_ELLIPTICAL_IF_ALL_CORRECT: ${TURN_OFF_ELLIPTICAL_IF_ALL_CORRECT}"
        echo "LOSS_AGG_MODE: ${LOSS_AGG_MODE}"
        echo "USE_KL_LOSS: ${USE_KL_LOSS}"
        echo "NORM_ADV_BY_STD_IN_GRPO: ${NORM_ADV_BY_STD_IN_GRPO}"
        echo "TURN_OFF_AT_HIGHEST_PASS_AT_K: ${TURN_OFF_AT_HIGHEST_PASS_AT_K}"
        echo "PASS_AT_K_FREQ: ${PASS_AT_K_FREQ}"
        echo "TRAIN_RANDOM_SUBSET_SIZE: ${TRAIN_RANDOM_SUBSET_SIZE}"
        echo "TRAIN_VAL_N: ${TRAIN_VAL_N}"
        echo "ALPHA: ${ALPHA}"
        echo "TEST_FREQ: ${TEST_FREQ}"
        sbatch --job-name=train_${ALGORITHM}_MATH_elliptical_${REWARD_MODEL_ENABLE}_beta_${BETA}_sparse_${SPARSE_DIM} scripts/train_grpo_math.slurm \
            ${MODEL_PATH} \
            ${REWARD_MODEL_ENABLE} \
            ${ELLIPTICAL_ENABLE} \
            ${SPARSE_DIM} \
            ${REWARD_MANAGER} \
            ${SEED} \
            ${BETA} \
            ${ROLLOUTS} \
            ${REWARD_TYPE} \
            ${RANDOMIZE_SPARSE_MATRIX} \
            ${TURN_OFF_ELLIPTICAL_IF_ANY_CORRECT} \
            ${TURN_OFF_ELLIPTICAL_IF_ALL_CORRECT} \
            ${LOSS_AGG_MODE} \
            ${USE_KL_LOSS} \
            ${NORM_ADV_BY_STD_IN_GRPO} \
            ${ALGORITHM} \
            ${TURN_OFF_AT_HIGHEST_PASS_AT_K} \
            ${PASS_AT_K_FREQ} \
            ${TRAIN_RANDOM_SUBSET_SIZE} \
            ${TRAIN_VAL_N} \
            ${ALPHA} \
            ${TEST_FREQ}
        echo "--------------------------------"
    done
done
