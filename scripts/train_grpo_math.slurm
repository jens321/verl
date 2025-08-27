#!/bin/bash
#SBATCH --job-name=train_grpo_MATH
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
REWARD_MODEL_ENABLE=$2
ELLIPTICAL_ENABLE=$3
SPARSE_DIM=$4
REWARD_MANAGER=$5
SEED=$6
BETA=$7
ROLLOUTS=$8
REWARD_TYPE=$9
RANDOMIZE_SPARSE_MATRIX=${10}
TURN_OFF_ELLIPTICAL_IF_NONE_CORRECT=${11}
TURN_OFF_ELLIPTICAL_IF_SOME_CORRECT=${12}
TURN_OFF_ELLIPTICAL_IF_ALL_CORRECT=${13}
LOSS_AGG_MODE=${14}
USE_KL_LOSS=${15}
NORM_ADV_BY_STD_IN_GRPO=${16}
ALGORITHM=${17}
TURN_OFF_AT_HIGHEST_PASS_AT_K=${18}
PASS_AT_K_FREQ=${19}
TRAIN_RANDOM_SUBSET_SIZE=${20}
TRAIN_VAL_N=${21}
ALPHA=${22}
TEST_FREQ=${23}
SAVE_FREQ=${24}
ELLIPTICAL_NORMALIZATION=${25}
RESUME_MODE=${26}
RESUME_FROM_PATH=${27}
PERSIST_COVARIANCE=${28}
KL_LOSS_COEF=${29}
TASK=${30}

math_train_path=$HOME/data/${TASK}/train.parquet
math_test_path=$HOME/data/${TASK}/test.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path']"

PYTHONUNBUFFERED=1 WANDB_MODE=offline TRANSFORMERS_OFFLINE=True WANDB_ENTITY=jtuyls python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.task=$TASK \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUTS \
    actor_rollout_ref.rollout.train_val_kwargs.n=$TRAIN_VAL_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=$REWARD_MODEL_ENABLE \
    reward_model.model.path="$MODEL_PATH" \
    reward_model.model.use_remove_padding=False \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=32 \
    reward_model.model.input_tokenizer=null \
    reward_model.elliptical.enable=$ELLIPTICAL_ENABLE \
    reward_model.elliptical.sparse_dim=$SPARSE_DIM \
    reward_model.elliptical.reward_type=$REWARD_TYPE \
    reward_model.elliptical.randomize_sparse_matrix=$RANDOMIZE_SPARSE_MATRIX \
    reward_model.elliptical.turn_off_at_highest_pass_at_k=$TURN_OFF_AT_HIGHEST_PASS_AT_K \
    reward_model.elliptical.normalization=$ELLIPTICAL_NORMALIZATION \
    reward_model.elliptical.persist_covariance=$PERSIST_COVARIANCE \
    reward_model.reward_manager=$REWARD_MANAGER \
    reward_model.reward_kwargs.beta=$BETA \
    reward_model.reward_kwargs.alpha=$ALPHA \
    reward_model.reward_kwargs.turn_off_elliptical_if_none_correct=$TURN_OFF_ELLIPTICAL_IF_NONE_CORRECT \
    reward_model.reward_kwargs.turn_off_elliptical_if_some_correct=$TURN_OFF_ELLIPTICAL_IF_SOME_CORRECT \
    reward_model.reward_kwargs.turn_off_elliptical_if_all_correct=$TURN_OFF_ELLIPTICAL_IF_ALL_CORRECT \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    algorithm.norm_adv_by_std_in_grpo=$NORM_ADV_BY_STD_IN_GRPO \
    algorithm.use_kl_in_reward=False \
    trainer.pass_at_k_freq=$PASS_AT_K_FREQ \
    trainer.train_random_subset_size=$TRAIN_RANDOM_SUBSET_SIZE \
    trainer.val_hard_subset=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='llm-exploration-rl-training' \
    trainer.experiment_name="${MODEL_PATH}_${ALGORITHM}_elliptical_${ELLIPTICAL_ENABLE}_beta_${BETA}_seed_${SEED}_sparse_${SPARSE_DIM}_reward_type_${REWARD_TYPE}_rollouts_${ROLLOUTS}_randomize_sparse_matrix_${RANDOMIZE_SPARSE_MATRIX}_turn_off_elliptical_if_none_correct_${TURN_OFF_ELLIPTICAL_IF_NONE_CORRECT}_turn_off_elliptical_if_some_correct_${TURN_OFF_ELLIPTICAL_IF_SOME_CORRECT}_turn_off_elliptical_if_all_correct_${TURN_OFF_ELLIPTICAL_IF_ALL_CORRECT}_train_val_n_${TRAIN_VAL_N}_turn_off_at_highest_pass_at_k_${TURN_OFF_AT_HIGHEST_PASS_AT_K}_pass_at_k_freq_${PASS_AT_K_FREQ}_train_random_subset_size_${TRAIN_RANDOM_SUBSET_SIZE}_alpha_${ALPHA}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=100 \
    trainer.seed=$SEED \
    trainer.resume_mode=$RESUME_MODE \
    trainer.resume_from_path=$RESUME_FROM_PATH

exit 0

# "${MODEL_PATH}_${ALGORITHM}_elliptical_${ELLIPTICAL_ENABLE}_beta_${BETA}_seed_${SEED}_sparse_${SPARSE_DIM}_reward_type_${REWARD_TYPE}_rollouts_${ROLLOUTS}_randomize_sparse_matrix_${RANDOMIZE_SPARSE_MATRIX}_turn_off_elliptical_if_none_correct_${TURN_OFF_ELLIPTICAL_IF_NONE_CORRECT}_turn_off_elliptical_if_some_correct_${TURN_OFF_ELLIPTICAL_IF_SOME_CORRECT}_turn_off_elliptical_if_all_correct_${TURN_OFF_ELLIPTICAL_IF_ALL_CORRECT}_train_val_n_${TRAIN_VAL_N}_turn_off_at_highest_pass_at_k_${TURN_OFF_AT_HIGHEST_PASS_AT_K}_pass_at_k_freq_${PASS_AT_K_FREQ}_train_random_subset_size_${TRAIN_RANDOM_SUBSET_SIZE}_alpha_${ALPHA}"