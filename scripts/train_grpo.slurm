#!/bin/bash
#SBATCH --job-name=train_grpo_default
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
ROLLOUTS=$3
LOSS_AGG_MODE=$4
USE_KL_LOSS=$5
NORM_ADV_BY_STD_IN_GRPO=$6
ALGORITHM=$7
TEST_FREQ=$8
SAVE_FREQ=$9
RESUME_MODE=${10}
RESUME_FROM_PATH=${11}
KL_LOSS_COEF=${12}
TASK=${13}
PPO_EPOCHS=${14}
SAVE_BEST_PASS_AT_1=${15}
SAVE_BEST_PASS_AT_64=${16}
CHECKPOINT_SAVE_CONTENTS=${17}
SAVE_BEST_HARD_PASS_AT_1=${18}
SAVE_BEST_HARD_PASS_AT_64=${19}
MAX_ACTOR_CKPT_TO_KEEP=${20}
TRAIN_BATCH_SIZE=${21}
PPO_MINI_BATCH_SIZE=${22}

math_train_path=$HOME/data/${TASK}/train.parquet
math_test_path=$HOME/data/${TASK}/dev.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path']"

PYTHONUNBUFFERED=1 WANDB_MODE=offline TRANSFORMERS_OFFLINE=True WANDB_ENTITY=jtuyls python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.task=$TASK \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.checkpoint.save_contents=$CHECKPOINT_SAVE_CONTENTS \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
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
    reward_model.model.path="$MODEL_PATH" \
    reward_model.model.use_remove_padding=False \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=32 \
    reward_model.model.input_tokenizer=null \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    algorithm.norm_adv_by_std_in_grpo=$NORM_ADV_BY_STD_IN_GRPO \
    algorithm.use_kl_in_reward=False \
    trainer.val_hard_subset=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='llm-exploration-rl-training' \
    trainer.experiment_name="${TASK}_grpo_seed_${SEED}_kl_${KL_LOSS_COEF}_ppo_epochs_${PPO_EPOCHS}" \
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