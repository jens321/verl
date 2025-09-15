#!/bin/bash
#SBATCH --job-name=train_unlikely
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=90
#SBATCH --mem=900G
#SBATCH --time=00-23:59:59    # Run for 23:59:59 hours
#SBATCH --gres=gpu:8
#SBATCH --partition=pli-c
#SBATCH --account=pli

set -x

MODEL_PATH=$1
SEED=$2
BETA=$3
ROLLOUTS=$4
LOSS_AGG_MODE=$5
USE_KL_LOSS=$6
NORM_ADV_BY_STD_IN_GRPO=$7
ALGORITHM=$8
TEST_FREQ=$9
SAVE_FREQ=${10}
RESUME_MODE=${11}
RESUME_FROM_PATH=${12}
KL_LOSS_COEF=${13}
TASK=${14}
SAVE_BEST_PASS_AT_1=${15}
SAVE_BEST_PASS_AT_64=${16}
CHECKPOINT_SAVE_CONTENTS=${17}
SAVE_BEST_HARD_PASS_AT_1=${18}
SAVE_BEST_HARD_PASS_AT_64=${19}
MAX_ACTOR_CKPT_TO_KEEP=${20}
TRAIN_BATCH_SIZE=${21}
PPO_MINI_BATCH_SIZE=${22}
DROP_SAMPLES_WITH_NO_ADV=${23}
PPO_EPOCHS=${24}
GEN_BATCH_SIZE=${25}
GRAD_SKIP_THRESH=${26}
TURN_OFF_UNLIKELY_IF_ALL_CORRECT=${27}
REMOVE_PREVIOUS_OPTIM_AND_EXTRA=${28}

EXTRA_GEN_ARG=""
if [ "$GEN_BATCH_SIZE" != "null" ]; then
  EXTRA_GEN_ARG="+data.gen_batch_size=${GEN_BATCH_SIZE}"
fi

EXTRA_GRAD_SKIP_THRESH_ARG=""
if [ "$GRAD_SKIP_THRESH" != "null" ]; then
  EXTRA_GRAD_SKIP_THRESH_ARG="actor_rollout_ref.actor.grad_skip_thresh=${GRAD_SKIP_THRESH}"
fi

# NOTE: unlikely repo sets rollout.n = 1, while we set it to 32
# therefore, we have to scale down the batch sizes that internally get upscaled
# by rollout.n
# PPO_MINI_BATCH_SIZE=$((256 / ${ROLLOUTS}))
# TRAIN_BATCH_SIZE=$((256 / ${ROLLOUTS}))

math_train_path=$HOME/data/${TASK}/train.parquet
math_test_path=$HOME/data/${TASK}/dev.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path']"

PYTHONUNBUFFERED=1 WANDB_MODE=offline TRANSFORMERS_OFFLINE=True WANDB_ENTITY=jtuyls python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.task=$TASK \
    data.drop_samples_with_no_adv=$DROP_SAMPLES_WITH_NO_ADV $EXTRA_GEN_ARG \
    actor_rollout_ref.actor.checkpoint.remove_previous_optim_and_extra=$REMOVE_PREVIOUS_OPTIM_AND_EXTRA \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.checkpoint.save_contents=$CHECKPOINT_SAVE_CONTENTS \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS $EXTRA_GRAD_SKIP_THRESH_ARG \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUTS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.model.use_remove_padding=False \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=32 \
    reward_model.model.input_tokenizer=null \
    reward_model.reward_manager=unlikely \
    reward_model.reward_kwargs.unlikely.beta=$BETA \
    reward_model.reward_kwargs.unlikely.turn_off_unlikely_if_all_correct=$TURN_OFF_UNLIKELY_IF_ALL_CORRECT \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    algorithm.norm_adv_by_std_in_grpo=$NORM_ADV_BY_STD_IN_GRPO \
    algorithm.use_kl_in_reward=False \
    trainer.val_hard_subset=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='llm-exploration-rl-training' \
    trainer.experiment_name="${TASK}_unlikely_seed_${SEED}_kl_${KL_LOSS_COEF}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=100 \
    trainer.seed=$SEED \
    trainer.resume_mode=$RESUME_MODE \
    trainer.resume_from_path=$RESUME_FROM_PATH \
    trainer.save_best_pass_at_1=$SAVE_BEST_PASS_AT_1 \
    trainer.save_best_pass_at_64=$SAVE_BEST_PASS_AT_64 \
    trainer.save_best_hard_pass_at_1=$SAVE_BEST_HARD_PASS_AT_1 \
    trainer.save_best_hard_pass_at_64=$SAVE_BEST_HARD_PASS_AT_64 \
    trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP

exit 0